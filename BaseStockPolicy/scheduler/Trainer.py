

class Trainer():
    def __init__(self, env, policies, policies_to_train, config):
        self.env = env
        self.step = 0
        self.policies = policies
        self.policies_to_train = policies_to_train
        self.training_steps = config['training_steps']
        self.batch_size = config['batch_size']
        self.update_period = config['update_period']

    def save(self, name, iter):
        agent_id = self.policies_to_train[0]
        policy = self.policies[agent_id]
        policy.save_param(name + f"_iter_{iter}")


    def train(self, iter, test=False, just_load_data=False):
        # TODO load validation data

        print(f"  == iteration {iter} == ")

        obss = self.env.reset()
        _, infos = self.env.state_calculator.world_to_state(self.env.world)
        rnn_states = {}
        rewards_all = {}
        episode_reward_all = {}
        episode_reward = {}
        policies_to_train_loss = {key : [] for key in self.policies_to_train}
        policies_to_train_qvalue = {key : [] for key in self.policies_to_train}
        policies_to_train_pred_loss =  {key : [] for key in self.policies_to_train}
        action_distribution = {key : [0 for i in range(int(self.env.action_space_consumer.n))] for key in self.policies_to_train}
        episode_steps = []
        episode_step = 0
        uncontrollable_part_state = {key: infos[key]['uncontrollable_part_state'].copy() for key in self.policies_to_train}
        uncontrollable_part_pred = {key: infos[key]['uncontrollable_part_pred'].copy() for key in self.policies_to_train}

        for agent_id in obss.keys():
            # policies[agent_id] = load_policy(agent_id)
            rnn_states[agent_id] = self.policies[agent_id].get_initial_state()
            rewards_all[agent_id] = []
            episode_reward_all[agent_id] = []
            episode_reward[agent_id] = 0

        for i in range(self.training_steps):
            self.step += 1
            episode_step += 1
            actions = {}
            # print("timestep : ", self.step)
            # print("Start calculate action ....")
            for agent_id, obs in obss.items():
                policy = self.policies[agent_id]
                action, new_state, _ = policy.compute_single_action(obs, state=rnn_states[agent_id],
                                                                    info=infos[agent_id],
                                                                    explore=True)
                actions[agent_id] = action
                # print(agent_id, " :", policy.__class__, " : ", action)
            next_obss, rewards, dones, infos = self.env.step(actions)

            for agent_id, reward in rewards.items():
                rewards_all[agent_id].append(reward)
                episode_reward[agent_id] += reward

            done = any(dones.values())

            for agent_id in self.policies_to_train:
                # print('policies_to_train: ', agent_id, " reward: ", rewards[agent_id])

                self.policies[agent_id].store_transition(obss[agent_id],
                                                         actions[agent_id],
                                                         rewards[agent_id],
                                                         next_obss[agent_id],
                                                         done,
                                                         uncontrollable_part_state[agent_id],
                                                         uncontrollable_part_pred[agent_id],
                                                         infos[agent_id]['uncontrollable_part_state'],
                                                         agent_id)
                uncontrollable_part_state[agent_id] = infos[agent_id]['uncontrollable_part_state'].copy()
                uncontrollable_part_pred[agent_id] = infos[agent_id]['uncontrollable_part_pred'].copy()
                action_distribution[agent_id][actions[agent_id]] += 1

            if self.step % (self.update_period * len(self.policies_to_train)) == 0:
                for agent_id in self.policies_to_train:
                    loss, qvalue, pred_loss = self.policies[agent_id].learn(self.batch_size)
                    policies_to_train_loss[agent_id].append(loss)
                    policies_to_train_qvalue[agent_id].append(qvalue)
                    policies_to_train_pred_loss[agent_id].append(pred_loss)
            if done:
                obss = self.env.reset()
                episode_steps.append(episode_step)
                episode_step = 0
                for agent_id, reward in episode_reward.items():
                    episode_reward_all[agent_id].append(reward)
                    episode_reward[agent_id] = 0
                if just_load_data:
                    break
            else:
                obss = next_obss
        infos = {
            "rewards_all": rewards_all,
            "episode_reward_all": episode_reward_all,
            "policies_to_train_loss": policies_to_train_loss,
            "policies_to_train_qvalue": policies_to_train_qvalue,
            "policies_to_train_pred_loss": policies_to_train_pred_loss,
            "action_distribution": action_distribution,
            "epsilon": self.policies[self.policies_to_train[0]].epsilon,
            "all_step": self.step,
            "episode_step": sum(episode_steps) / len(episode_steps),
        }
        # dqn_policy = self.policies[self.policies_to_train[0]]
        # print('random action: ', dqn_policy.rand_action, ' greedy action:', dqn_policy.greedy_action)
        # dqn_policy.rand_action = 0
        # dqn_policy.greedy_action = 0
        return infos




