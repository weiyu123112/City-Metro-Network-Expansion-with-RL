"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import sys
sys.path.append('/home/weiyu/program/metro_expand_combination/att3/')

from metro_model import DRL4Metro, Encoder
import metro_vrp


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# print(device)


class StateCritic(nn.Module): # ststic+ dynamic + vector present
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum()
        return output

class StateCritic1(nn.Module): # ststic+ dynamic + matrix present
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic1, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv2d(hidden_size * 2, 20, kernel_size=5, stride=1, padding=2)
        self.fc2 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2)
        self.fc3 = nn.Linear(20 * args.grid_x_max * args.grid_y_max, 1)

        #
        # self.fc3 = nn.Linear(20 * args.grid_x_max * args.grid_y_max, 36)
        # self.fc4 = nn.Linear(36, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, hidden_size, grid_x_max, grid_y_max):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        static_hidden = static_hidden.view(hidden_size, grid_x_max, grid_y_max)
        dynamic_hidden = dynamic_hidden.view(hidden_size, grid_x_max, grid_y_max)

        hidden = torch.cat((static_hidden, dynamic_hidden), 0).unsqueeze(0)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = output.view(output.size(0), -1)
        output = self.fc3(output)
        # output = self.fc4(output)
        return output


class Critic(nn.Module): # only dynamic0 + vector present
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, dynamic_size, hidden_size):
        super(Critic, self).__init__()

        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dynamic):

        dynamic_hidden = self.dynamic_encoder(dynamic)

        output = F.relu(self.fc1(dynamic_hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic1(nn.Module): # only dynamic0 + matrix present
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, dynamic_size, hidden_size):
        super(Critic1, self).__init__()

        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv2d(hidden_size, 20, kernel_size=5, stride=1, padding=2)
        self.fc2 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2)
        self.fc3 = nn.Linear(20 * args.grid_x_max * args.grid_y_max, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dynamic, hidden_size, grid_x_max, grid_y_max):

        dynamic_hidden = self.dynamic_encoder(dynamic)

        dynamic_hidden = dynamic_hidden.view(hidden_size, grid_x_max, grid_y_max).unsqueeze(0)

        output = F.relu(self.fc1(dynamic_hidden))
        output = F.relu(self.fc2(output))
        output = output.view(output.size(0), -1)
        output = self.fc3(output)
        return output


def train(actor, critic, train_data, reward_fn,
         epoch_max, actor_lr, critic_lr, max_grad_norm, result_path, od_index_path, train_size, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(result_path, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    average_reward_list, actor_loss_list, critic_loss_list, average_od_list, average_Ac_list = [], [], [], [], []

    # best_params = None
    best_reward = 0

    static, dynamic = train_data.static, train_data.dynamic
    grid_num = train_data.grid_num
    exist_line_num = train_data.exist_line_num
    line_full_tensor = train_data.line_full_tensor
    line_station_list = train_data.line_station_list

    # od_index_path = r'/home/weiyu/program/metro_expand_combination/od_index.txt'
    # od_matirx = metro_vrp.build_od_matrix(grid_num, od_index_path)  #GPU need
    od_matirx = metro_vrp.build_od_matrix1(grid_num, od_index_path)   #CPU need
    od_matirx =  od_matirx / torch.max(od_matirx)                     #GPU and CPU need
    # od_matirx = od_matirx.half()  # turn to float16 to recude CUDA memory  GPU need

    # exclude the needed od pair
    exclude_pair = metro_vrp.exlude_od_pair(args.grid_x_max)
    od_matirx = metro_vrp.od_matrix_exclude(od_matirx, exclude_pair)

    if args.social_equity:
    # path_house = r'/home/weiyu/program/metro_expand_combination/index_average_price.txt'
        price_matrix = metro_vrp.build_grid_price(args.path_house, args.grid_x_max, args.grid_y_max)
        price_matrix = price_matrix / torch.max(price_matrix)
        # price_matrix = price_matrix.half() # turn to float16 to recude CUDA memory : GPU

    if args.initial_direct:
        direction_list = args.initial_direct.split(',')
        initial_direct = []

        for i in direction_list:
            initial_direct.append(int(i))
    else:
        initial_direct = None

    # epoch_max = 2000

    for epoch in range(epoch_max):

        actor.train()
        critic.train()

        epoch_start = time.time()
        start = epoch_start

        od_list, social_equity_list = [], []

        for example_id in range(train_size):  # this loop accumulates a batch
            #once
            tour_idx, tour_logp = actor(static, dynamic, args.station_num_lim, args.budget, initial_direct,
                                        args.line_unit_price, args.station_price, decoder_input=None, last_hh=None)

            tour_idx_cpu = tour_idx.cpu()
            tour_idx_np = tour_idx_cpu.numpy()
            agent_grid_list = tour_idx_np[0].tolist()

            # Sum the log probabilities for each city in the tour
            if args.social_equity == 1:
                reward_od = metro_vrp.reward_fn1(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list,
                                              exist_line_num, od_matirx, args.grid_x_max, args.dis_lim)  #CPU
                agent_Ac = metro_vrp.agent_grids_price(tour_idx_cpu, args.grid_x_max, price_matrix) #cpu

                od_list.append(reward_od.item())
                social_equity_list.append(agent_Ac.item())

                reward = args.factor_weight*reward_od + (1-args.factor_weight)*agent_Ac
                reward = reward.to(device)

            elif args.social_equity == 2:
                reward_od = metro_vrp.reward_fn1(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list,
                                              exist_line_num, od_matirx, args.grid_x_max, args.dis_lim)  # CPU
                agent_Ac = metro_vrp.agent_grids_price1(tour_idx_cpu, args.grid_x_max, price_matrix)  # cpu

                od_list.append(reward_od.item())
                social_equity_list.append(agent_Ac.item())

                reward = args.factor_weight1*reward_od - (1 - args.factor_weight1)*agent_Ac

                # reward = args.factor_weight1 * reward_od - 0.5 * agent_Ac
                reward = reward.to(device)

            else:
                reward = metro_vrp.reward_fn1(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list,
                                      exist_line_num, od_matirx, args.grid_x_max, args.dis_lim)

                od_list.append(reward.item())
                social_equity_list.append(0)
                reward = reward.to(device)



            # Query the critic for an estimate of the reward

            # critic_est = critic(static, dynamic).view(-1)   # ststic+ dynamic + vector present

            critic_est = critic(static, dynamic, args.hidden_size, args.grid_x_max, args.grid_y_max).view(-1)  # ststic+ dynamic + matrix present
            # critic_est = critic(static, dynamic, args.hidden_size, args.grid_x_max, args.grid_y_max).view(-1)

            # critic_est = critic(dynamic).view(-1)             # only dynamic + vector present

            # critic_est = critic(dynamic, args.hidden_size, args.grid_x_max, args.grid_y_max).view(-1)          # only dynamic + matrix present


            advantage = (reward - critic_est)
            per_actor_loss = -advantage.detach()*tour_logp.sum(dim=1)
            per_critic_loss = advantage ** 2

            if example_id == 0:
                actor_loss0 = per_actor_loss
                critic_loss0 = per_critic_loss
                rewards = reward
            else:
                actor_loss0 = actor_loss0 + per_actor_loss
                critic_loss0 = critic_loss0 + per_critic_loss
                rewards = rewards + reward

        actor_loss = actor_loss0 / train_size
        critic_loss = critic_loss0 / train_size
        average_reward = rewards / train_size
        average_od = sum(od_list)/len(od_list)
        average_Ac = sum(social_equity_list)/len(social_equity_list)

        average_reward_list.append(average_reward.half().item())
        actor_loss_list.append(actor_loss.half().item())
        critic_loss_list.append(critic_loss.half().item())
        average_od_list.append(average_od)
        average_Ac_list.append(average_Ac)

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_optim.step()

        end = time.time()
        cost_time = end - start
        print('epoch %d, average_reward: %2.3f, actor_loss: %2.4f,  critic_loss: %2.4f, cost_time: %2.4fs'
              % (epoch, average_reward.item(), actor_loss.item(), critic_loss.item(), cost_time))

        torch.cuda.empty_cache() # reduce memory
        ########  finish an updata with a batch


        # Save the weights of an epoch
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save best model parameters
        average_reward_value = average_reward.item()
        if average_reward_value > best_reward:
            best_reward = average_reward_value

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

    records_path = os.path.join(save_dir, 'reward_actloss_criloss.txt')

    write_file = open(records_path, 'w')

    for i in range(epoch_max):
        per_average_reward_record = average_reward_list[i]
        per__actor_loss_record = actor_loss_list[i]
        per_critic_loss_record = critic_loss_list[i]
        per_epoch_od = average_od_list[i]
        per_epoch_Ac = average_Ac_list[i]

        to_write = str(per_average_reward_record) +'\t' + str(per__actor_loss_record) + '\t'+ str(per_critic_loss_record) + '\t'+ str(per_epoch_od) + '\t' + str(per_epoch_Ac) + '\n'

        write_file.write(to_write)
    write_file.close()

    picture_path = os.path.join(save_dir, 'loss.png')
    # plt.subplot(2, 1, 1)
    plt.plot(average_reward_list, '-', label="reward")
    plt.title('Reward vs. epoches')
    plt.ylabel('Reward')
    plt.legend(loc='best')

    # plt.subplot(2, 1, 2)
    # plt.plot(critic_loss_list, 'o-', label="critic_loss")
    # plt.xlabel('Critic_loss vs. epoches')
    # plt.ylabel('Critic loss')
    # plt.legend(loc='best')
    plt.savefig(picture_path, dpi=800)








def train_vrp(args):


    import metro_vrp
    from metro_vrp import MetroDataset

    STATIC_SIZE = 2
    DYNAMIC_SIZE = 1

    if args.initial_station:
        ini_x, ini_y = args.initial_station.split(',')
        initial_station = []

        initial_station.append(int(ini_x))
        initial_station.append(int(ini_y))
    else:
        initial_station = None

    train_data = MetroDataset(args.grid_x_max, args.grid_y_max, args.exist_line_num, initial_station, static_size=2, dynamic_size=1)

    actor = DRL4Metro(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    train_data.v_to_g,
                    train_data.vector_allow,
                    args.num_layers,
                    args.dropout).to(device)


    # critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)  # ststic+ dynamic0 + vector present

    critic = StateCritic1(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)   # ststic+ dynamic0 + matrix present

    # critic = Critic(DYNAMIC_SIZE, args.hidden_size).to(device)                    # only dynamic0 + vector present

    # critic = Critic1(DYNAMIC_SIZE, args.hidden_size).to(device)                   # only dynamic0 + matrix present

    kwargs = vars(args)  # dict

    kwargs['train_data'] = train_data
    kwargs['reward_fn'] = metro_vrp.reward_fn


    if args.checkpoint:  # test: give model_solution
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

        static, dynamic = train_data.static, train_data.dynamic

        tour_idx, tour_logp, dynamic0 = actor(static, dynamic, decoder_input=None, last_hh=None)

        result_time = '%s' % datetime.datetime.now().time()
        result_time = result_time.replace(':', '_')
        model_solution_path = os.path.join(args.result_path, result_time, 'tour_idx.txt')

        f = open(model_solution_path, 'w')

        to_write = ''
        for i in tour_idx[0]:
            to_write = to_write + str(i.item()) + ','

        to_write1 = to_write.rstrip(',')
        f.write(to_write1)
        f.close()


    if not args.test:  # train
        train(actor, critic, **kwargs)







def model_solution(grid_x_max, grid_y_max, exist_line_num, hidden_size,path, initial_station, initial_direct,station_num_lim,
                   budget, line_unit_price, station_price):
    # grid_x_max, grid_y_max = 29, 29
    # hidden_size =
    # path = r'/home/weiyu/program/metro_expand_combination/result/21_04_10.890831/actor.pt'

    from metro_vrp import MetroDataset

    STATIC_SIZE = 2
    DYNAMIC_SIZE = 1


    train_data = MetroDataset(grid_x_max, grid_y_max, exist_line_num, initial_station, static_size=2, dynamic_size=1)

    static = train_data.static
    dynamic = train_data.dynamic

    model = DRL4Metro(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    train_data.v_to_g,
                    train_data.vector_allow,
                    num_layers=1,
                    dropout=0.).to(device)


    model.load_state_dict(torch.load(path))
    model.eval()   
    # model.train() 

    tour_idx, tour_logp = model(static, dynamic, station_num_lim, budget,
                                initial_direct, line_unit_price, station_price, decoder_input=None, last_hh=None)

    print('tour_idx:', tour_idx)






if __name__ == '__main__':

     parser = argparse.ArgumentParser(description='Combinatorial Optimization')
     parser.add_argument('--grid_x_max', default=29, type=int)
     parser.add_argument('--grid_y_max', default=29, type=int)
    
     parser.add_argument('--checkpoint', default=None)
     
    
     parser.add_argument('--test', action='store_true', default=False)
    
     parser.add_argument('--actor_lr', default=5e-4, type=float)
     parser.add_argument('--critic_lr', default=5e-4, type=float)
     parser.add_argument('--max_grad_norm', default=2., type=float)
     parser.add_argument('--batch_size', default=1, type=int)
     parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
     parser.add_argument('--dropout', default=0.1, type=float)
     parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    
     parser.add_argument('--train_size',default=128, type=int)   # similar to batch size
    
     parser.add_argument('--exist_line_num', default=2, type=int)
     parser.add_argument('--epoch_max', default=3500, type=int) # the number of total epoch
    
     parser.add_argument('--initial_station', default=None)
     # default=None when there is no initial station; otherwise example: default ='14,2', type = str
     parser.add_argument('--initial_direct', default=None)
     # default=None when there is no initial direct, otherwise example: default ='0,2', type = str
    
     parser.add_argument('--station_num_lim', default=45, type=int)  # limit the number of stations in a line
     parser.add_argument('--budget', default=None)
     # if budget = None, there is no cost limit.
     # budget example:  default=200, type=int
     parser.add_argument('--line_unit_price', default=1.0, type=float)
     parser.add_argument('--station_price', default=5.0, type=float)
    
     parser.add_argument('--dis_lim', default=None)
     #example1:  '--dis_lim', default=-1, type=int
     #            od pairs in reward only consider agent line
     #example2:  '--dis_lim', default=None
     #            agent station和existing statio
     #example3:  '--dis_lim', default=2.0, type=float
     # #           
    
    
     parser.add_argument('--social_equity', default=1, type=int)
     parser.add_argument('--factor_weight', default=1.0, type=float)
     parser.add_argument('--factor_weight1', default=0.5, type=float)
     # if social_equity= 0, reward does not contain the social equality
     # if social_equity= 1, reward contains the first social equity: utilitarianism
     #                             reward = factor_weight * reward_od + (1 - factor_weight) * agent_Ac
     # if social_equity= 2, reward contains the second social equity: equal sharing
     #                             reward = factor_weight1*reward_od - (1 - factor_weight1)*agent_Ac
     
     parser.add_argument('--result_path', default='/home/weiyu/program/metro_expand_combination/att3/result/', type=str)
     parser.add_argument('--od_index_path', default='/home/weiyu/program/metro_expand_combination/att3/od_index.txt', type=str)
     parser.add_argument('--path_house', default='/home/weiyu/program/metro_expand_combination/att3/index_average_price.txt', type=str)
    
     args = parser.parse_args()
    
     train_vrp(args)



################################################
#    # /checkpoints/1500/   att3_210_1
#    #
#    path = r'/home/weiyu/program/metro_expand_combination/att3/result/att3_210_3/checkpoints/3390/actor.pt'
#    # path_attn = r'/home/weiyu/program/metro_expand_combination/att3/result/att3_5/attn.txt'
#
#    grid_x_max, grid_y_max = 29, 29
#    hidden_size = 128
#                         #############
#    exist_line_num = 2
#
#    initial_station = None
#    initial_direct = None
#
#    # initial_station =[28,2]
#    # initial_direct=[2,4]
#
#    station_num_lim = 35
#    budget = 210
#    line_unit_price = 1.0
#    station_price = 5.0
#
#
#    model_solution(grid_x_max, grid_y_max, exist_line_num, hidden_size,path, initial_station, initial_direct,station_num_lim,
#                   budget, line_unit_price, station_price)
##










