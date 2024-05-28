import os
import sys
import random
import argparse

import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm

def train(train_dataloader, save_dir, save_model_every, \
        text_encoder, image_decoder, noise_scheduler, \
        scheduled_sampling_weights_start, scheduled_sampling_weights_end, \
        optimizer, lr_scheduler, num_epochs, gradient_accumulation_steps=1, \
        clip_grad_norm=1.0, learning_rate=1e-4, \
        mixed_precision='no'):
    
    #what is this?
    learning_rate = optimizer.defaults['lr']

    """
    Accelerator class is part of Hugging Face's accelerate library. It simplifies 
    Pytorch operations on multiple GPUs

    mixed precision: using a combination of 32 and 16 bit floting pointy types to do computation
    16-bit aka half precision which using less memory bandwith allows more data to be prcoess -> trainign vroom vroom
    this code uses only 32 bit more memory and takes longer idk why mixed not used seems like a clear cut better option


    gradient accumualtion step: why am I limiting the number of steps I am collecting gradient
    """

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        logging_dir=os.path.join(save_dir, "logs")
    )
    #just move to the GPU lor
    text_encoder = text_encoder.to(accelerator.device)

    #is this cause they want the process in the main gpu to finsh before startgn others?
    if accelerator.is_main_process:
        accelerator.init_trackers("markup2im_train")
    
    #sort of config the gpus for distrubuted training? helps with model splitting etc
    image_decoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        image_decoder, optimizer, train_dataloader, lr_scheduler
    )



    """
    scheduled_sampling_weights_start: probability distribution over different 
    levels of model reliance aka 
    how much the model relies on its own predictions)

    scheduled_sampling_weights_end:probability distribution over the same levels 
    at the end of training
    shifting towards higher reliance on the model's own outputs. 

    based on 0% to 50%?
    """
    scheduled_sampling_weights_start = np.array(scheduled_sampling_weights_start)
    scheduled_sampling_weights_end = np.array(scheduled_sampling_weights_end)


    global_step = 0
    total_steps = len(train_dataloader)*num_epochs
    for epoch in range(num_epochs):
        """
        this is a progress bar lol. like total/batch?
        accelerator.is_local_main_process: returns true if current process is on main gpu?
        bdisable=not accelerator.is_local_main_process: basically you dont want to have a gazzllion progress bars this means
        progres bars only appears when you are doing something on main gpu
        """
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        # to set the progress bar to reflect the epoch count
        progress_bar.set_description(f"Epoch {epoch}")

        """
        this is a batch processing loop. iterates through each batch of data
        """
        for step, batch in enumerate(train_dataloader):
            """
            global_step / total_steps : proportion of the total training that has been completed
            i think it is to adjust the 0 to 50% use of sheduched sampling
            """
            m_probs = scheduled_sampling_weights_start + global_step / total_steps * \
                    (scheduled_sampling_weights_end - scheduled_sampling_weights_start)
            
            """
            np.insert(m_probs, 0, 1-m_probs.sum()) is used to insert a new value at the start of the 
            m_probs array. The value inserted is 1-m_probs.sum(), 
            which represents the probability of using the true data directly 
            instead of any predicted or diffused data
            sort of like teacher forcing I guess?
            """
            m_probs = np.insert(m_probs, 0, 1-m_probs.sum()) 
            disp_str = '"' + ' '.join([f'm={i} ({m_probs[i]:.2f})' for i in range(len(m_probs))]) + '"'

            """
            batch['images']: retrieves OG/no noise image corresponding to
            current batch

            batch['input_ids']: lol. seems NLP like but idk. send halp

            masks: used to incorporate attention mechanisms trasnfromer type beat

            all the to(accelerator.device): just moves the data to the main GPU? to help
            with parrallel compute

            encoder_hidden_states: send halp

            bs: retrives batch size. first dimension of cleam_images. why isnt this preset?
            """
            clean_images = batch['images'].to(accelerator.device)
            input_ids = batch['input_ids'].to(accelerator.device)
            masks = batch['attention_mask'].to(accelerator.device)
            encoder_hidden_states = encode_text(text_encoder, input_ids, masks)
            bs = clean_images.shape[0]

            """
            take a random timestamp 
            and introduce gaussian noise to get to that particular noised image at that timestamp
            cause gaussian distribution is closed under convolution
            """
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            """
            this randomly chooses how many "loops" of sheducled sampling we will go through
            aka how many timestamps from the randomly chosen timestamp above the model simulate/predict
            """
            m = np.random.choice(len(m_probs), size=1, p=m_probs)[0]

            """
            this is to adjust for invalid m values lets say I randomly pick timestamp 2 and m is 5 how can?
            so I adjust till I can actually move m steps
            """
            while max(timesteps) >= noise_scheduler.num_train_timesteps-m:
                m -= 1

            """
            torch.no_grad: gradient isnt being accumualted
            """
            with torch.no_grad():
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                # first, sample t + m using Q
                noisy_images_t_plus_m = noise_scheduler.add_noise(clean_images, noise, timesteps+m)
                noisy_images_t_plus_s = noisy_images_t_plus_m
                # next, roll back to t using P
                for s in range(m):
                    # predict noise
                    noise_pred_rollback_s = image_decoder(noisy_images_t_plus_s, timesteps+m-s, encoder_hidden_states, attention_mask=masks)["sample"]
                    lambs_s, alpha_prod_ts_s = noise_scheduler.get_lambda_and_alpha(timesteps+m-s)
                    # clean img predicted
                    x_0_pred = (noisy_images_t_plus_s - lambs_s.view(-1, 1, 1, 1) * noise_pred_rollback_s) / alpha_prod_ts_s.view(-1, 1, 1, 1)
                    noise = torch.randn(clean_images.shape).to(clean_images.device)
                    # get previous step sample
                    noisy_images_t_plus_s_minus_one  = noise_scheduler.add_noise(x_0_pred, noise, timesteps + m-s-1)
                    # update
                    noisy_images_t_plus_s = noisy_images_t_plus_s_minus_one
                noisy_images_t = noisy_images_t_plus_s

            with accelerator.accumulate(image_decoder):
                # Predict the noise residual
                noise_pred = image_decoder(noisy_images_t, timesteps, encoder_hidden_states, attention_mask=masks)["sample"]
                lambs_t, alpha_prod_ts_t = noise_scheduler.get_lambda_and_alpha(timesteps)
                noise = (noisy_images_t - alpha_prod_ts_t.view(-1, 1, 1, 1) * clean_images) / lambs_t.view(-1, 1, 1, 1)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(image_decoder.parameters(), clip_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()*gradient_accumulation_steps, "lr": lr_scheduler.get_last_lr()[0], "step": global_step, 'scheduled sampling': disp_str}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        if epoch % save_model_every == 0:
            save_model(image_decoder, os.path.join(save_dir, f'model_e{num_epochs}_lr{learning_rate}.pt.{epoch}'))

def main(args):
    
    # Check arguments
    assert len(args.scheduled_sampling_weights_start) == len(args.scheduled_sampling_weights_end)
    assert all([0 <= item <= 1 for item in args.scheduled_sampling_weights_start])
    assert all([0 <= item <= 1 for item in args.scheduled_sampling_weights_end])
    assert sum(args.scheduled_sampling_weights_start) <= 1
    assert sum(args.scheduled_sampling_weights_end) <= 1
    # Get default arguments
    if (args.image_height is not None) and (args.image_width is not None):
        image_size = (args.image_height, args.image_width)
    else:
        print (f'Using default image size for dataset {args.dataset_name}')
        image_size = get_image_size(args.dataset_name)
        print (f'Default image size: {image_size}')
    args.image_size = image_size
    if args.input_field is not None:
        input_field = args.input_field
    else:
        print (f'Using default input field for dataset {args.dataset_name}')
        input_field = get_input_field(args.dataset_name)
        print (f'Default input field: {input_field}')
    args.input_field = input_field
    if args.encoder_model_type is not None:
        encoder_model_type = args.encoder_model_type
    else:
        print (f'Using default encoder model type for dataset {args.dataset_name}')
        encoder_model_type = get_encoder_model_type(args.dataset_name)
        print (f'Default encoder model type: {encoder_model_type}')
    args.encoder_model_type = encoder_model_type
    if args.color_mode is not None:
        color_mode = args.color_mode
    else:
        print (f'Using default color mode for dataset {args.dataset_name}')
        color_mode = get_color_mode(args.dataset_name)
        print (f'Default color mode: {color_mode}')
    args.color_mode = color_mode 
    assert args.color_mode in ['grayscale', 'rgb']
    if args.color_mode == 'grayscale':
        args.color_channels = 1
    else:
        args.color_channels = 3

    # Load data
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.shuffle(seed=args.seed1)
   
    # Load input tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_type)

    # Load input encoder
    text_encoder = AutoModel.from_pretrained(args.encoder_model_type).cuda()
  
    # Preprocess data to form batches
    transform_list = []
    if args.color_mode == 'grayscale':
        transform_list.append(transforms.Grayscale(num_output_channels=args.color_channels))
    preprocess_image = transforms.Compose(
        transform_list + [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    def preprocess_formula(formula):
        example = tokenizer(formula, truncation=True, max_length=args.max_input_length)
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        return input_ids, attention_mask
    
    def transform(examples):
        images = [preprocess_image(image.convert("RGB")) for image in examples["image"]]
        gold_images = [image for image in examples["image"]]
        formulas_and_masks = [preprocess_formula(formula) for formula in examples[args.input_field]]
        formulas = [item[0] for item in formulas_and_masks]
        masks = [item[1] for item in formulas_and_masks]
        filenames = examples['filename']
        return {'images': images, 'input_ids': formulas, 'attention_mask': masks, 'filenames': filenames, 'gold_images': gold_images}
    
    dataset.set_transform(transform)

    def collate_fn(examples):
        eos_id = tokenizer.encode(tokenizer.eos_token)[0] # legacy code, might be unnecessary
        max_len = max([len(example['input_ids']) for example in examples]) + 1
        examples_out = []
        for example in examples:
            example_out = {}
            orig_len = len(example['input_ids'])
            formula = example['input_ids'] + [eos_id,] * (max_len - orig_len)
            example_out['input_ids'] = torch.LongTensor(formula)
            attention_mask = example['attention_mask'] + [1,] + [0,] * (max_len - orig_len - 1)
            example_out['attention_mask'] = torch.LongTensor(attention_mask)
            example_out['images'] = example['images']
            examples_out.append(example_out)
        batch = default_collate(examples_out)
        filenames = [example['filenames'] for example in examples]
        gold_images = [example['gold_images'] for example in examples]
        batch['filenames'] = filenames
        batch['gold_images'] = gold_images 
        return batch
    
    torch.manual_seed(args.seed2)
    random.seed(args.seed2)
    np.random.seed(args.seed2)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, \
            shuffle=True, collate_fn=collate_fn, worker_init_fn=np.random.seed(0), \
            num_workers=args.num_dataloader_workers)

    # Create and load models
    text_encoder = AutoModel.from_pretrained(args.encoder_model_type).cuda()
    # forward a fake batch to figure out cross_attention_dim
    hidden_states = encode_text(text_encoder, torch.zeros(1,1).long().cuda(), None)
    cross_attention_dim = hidden_states.shape[-1]

    image_decoder = create_image_decoder(image_size=args.image_size, color_channels=args.color_channels, \
            cross_attention_dim=cross_attention_dim) 
    image_decoder = image_decoder.cuda()

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")
    # Optimization
    optimizer = torch.optim.AdamW(image_decoder.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    train(train_dataloader, args.save_dir, args.save_model_every, \
        text_encoder, image_decoder, noise_scheduler, \
        args.scheduled_sampling_weights_start, args.scheduled_sampling_weights_end, \
        optimizer, lr_scheduler, args.num_epochs, \
        gradient_accumulation_steps=args.gradient_accumulation_steps, \
        clip_grad_norm=args.clip_grad_norm, \
        mixed_precision=args.mixed_precision)

    # Save final model
    save_model(image_decoder, os.path.join(args.save_dir, f'model_e{args.num_epochs}_lr{args.learning_rate}.pt.{args.num_epochs}'))


if __name__ == '__main__':
    args = process_args(sys.argv[1:])
    main(args)