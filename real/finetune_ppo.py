import os
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import torch
import wandb

from agent.ppo_agent import FinetunePPO

from logger import Logger
from tqdm import tqdm
from dataset.ppo_dataset import prepare_real_pen_data_ppo, set_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(agent, validation_loader, L, epoch):
    loss_val = 0
    with torch.inference_mode():
        agent.set_eval()
        for iter, data_batch in enumerate(validation_loader):
            obs, proprio_hist, finger_tip_pos, action = data_batch
            loss = agent.evaluate(
                obs.cuda(), proprio_hist.cuda(), finger_tip_pos.cuda(), action.cuda())
            loss_val += loss

    loss_val /= len(validation_loader)

    return loss_val

################## compute loss in one iteration #######################


def compute_loss(agent, bc_train_dataloader, L, epoch):

    data_batch = next(iter(bc_train_dataloader))
    obs, proprio_hist, finger_tip_pos, action = data_batch
    loss_dict = agent.compute_loss(
        obs.cuda(), proprio_hist.cuda(), finger_tip_pos.cuda(), action.cuda())

    return loss_dict


def train_in_one_epoch(agent, it_per_epoch, bc_train_dataloader, bc_validation_dataloader, L, epoch, sim_real_ratio=1):

    loss_train = 0
    with torch.enable_grad():
        for _ in tqdm(range(it_per_epoch)):

            l1_loss = compute_loss(agent, bc_train_dataloader, L, epoch)
            loss = agent.update_policy(l1_loss)
            loss_train += loss

    loss_val = evaluate(agent, bc_validation_dataloader, L, epoch)
    agent.set_train()
    loss_train_dict = {
        "train": loss_train/(it_per_epoch),
        "val": loss_val
    }

    return loss_train_dict


def main(args):
    # read and prepare data
    set_seed(args["seed"])
   
    Prepared_Data = prepare_real_pen_data_ppo(args['real_dataset_folder'], args['real_batch_size'],
                                            args['val_ratio'], seed=args['seed'])
    print('Data prepared')
   
    bc_train_set = Prepared_Data['bc_train_set']

    concatenated_obs_shape = bc_train_set.dummy_data['obs'].shape
    print("Concatenated Observation (State + Visual Obs) Shape: {}".format(concatenated_obs_shape))
    action_shape = bc_train_set.dummy_data['action'].shape
    print("Action shape: {}".format(action_shape))
    # make agent
    agent = FinetunePPO(args)
    agent.load(args['checkpoint_path'])
    L = Logger("{}_{}".format(args['input_mode'], args['num_epochs']))
   
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"{args['real_dataset_folder']}_{cur_time}")
    wandb.init(
        project="dex-pen-finetune-ppo",
        name=os.path.basename(log_dir),
        config=args
    )
    os.makedirs(log_dir, exist_ok=True)

    best_success = 0

    for epoch in range(args['num_epochs']):
        print('  ', 'Epoch: ', epoch)
        agent.set_train()
      
        loss_train_dict = train_in_one_epoch(agent, Prepared_Data['it_per_epoch'], Prepared_Data['bc_train_dataloader'],
                                                Prepared_Data['bc_validation_dataloader'], L, epoch)
        metrics = {
            "loss/train": loss_train_dict["train"],
            "loss/val": loss_train_dict["val"],
            "epoch": epoch
        }

        if (epoch + 1) % args["eval_freq"] == 0 and (epoch+1) >= args["eval_start_epoch"]:
            # total_steps = x_steps * y_steps = 4 * 5 = 20
            agent.save(os.path.join(
                log_dir, f"epoch_{epoch + 1}.pt"))

        wandb.log(metrics)

    wandb.finish()



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--real-dataset-folder", default=None, type=str)
    parser.add_argument("--checkpoint-path", default=None, type=str)
    parser.add_argument("--real-batch-size", default=8192, type=int)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num-epochs", default=1000, type=int)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--eval-freq", default=100, type=int)
    parser.add_argument("--eval-start-epoch", default=400, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--input-mode", default='proprio', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    args = {
        'real_dataset_folder': args.real_dataset_folder,
        'checkpoint_path': args.checkpoint_path,
        # 8192 16384 32678 65536
        'real_batch_size': args.real_batch_size,
        'val_ratio': args.val_ratio,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'weight_decay': args.weight_decay,
        "eval_freq": args.eval_freq,
        "eval_start_epoch": args.eval_start_epoch,
        "seed": args.seed,
        "input_mode": args.input_mode
    }

    main(args)
