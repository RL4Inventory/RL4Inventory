import global_config
from utility.tools import SimulationTracker
from env.inventory_utils import Utils

class Trainer():
    def __init__(self, env, policies, policies_to_train, config):
        self.env = env
        self.step = 0
        self.policies = policies
        self.policies_to_train = policies_to_train
        self.training_steps = config['training_steps']
        # self.eval_steps = config['eval_steps']
        self.batch_size = config['batch_size']
        self.update_period = config['update_period']
        self.uncontrollable_part_state_key = 'uncontrollable_part_state'
        if global_config.use_cnn_state:
            self.uncontrollable_part_state_key = 'uncontrollable_part_state_cnn'

    def save(self, name, iter):
        agent_id = self.policies_to_train[0]
        policy = self.policies[agent_id]
        policy.save_param(name + f"_iter_{iter}")

    def switch_mode(self, eval):  # only for epsilon greedy
        for policy in self.policies_to_train:
            self.policies[policy].switch_mode(eval=eval)


    def train(self, iter):
        self.switch_mode(eval=False)
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
        uncontrollable_part_state = {key: infos[key][self.uncontrollable_part_state_key].copy() for key in self.policies_to_train}
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
                                                         infos[agent_id][self.uncontrollable_part_state_key],
                                                         agent_id)
                uncontrollable_part_state[agent_id] = infos[agent_id][self.uncontrollable_part_state_key].copy()
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

    def eval(self, iter, eval_on_trainingset=False):
        self.switch_mode(eval=True)

        print(f"  == eval iteration {iter} == ")

        obss = self.env.reset(eval=True, eval_on_trainingset=eval_on_trainingset)
        _, infos = self.env.state_calculator.world_to_state(self.env.world)
        rnn_states = {}
        rewards_all = {}
        episode_reward_all = {}
        episode_reward = {}
        episode_steps = []
        episode_step = 0

        tracker = SimulationTracker(self.env.done_step, 1, self.env.agent_ids())

        for agent_id in obss.keys():
            # policies[agent_id] = load_policy(agent_id)
            rnn_states[agent_id] = self.policies[agent_id].get_initial_state()
            rewards_all[agent_id] = []
            episode_reward_all[agent_id] = []
            episode_reward[agent_id] = 0

        for i in range(100000):
            episode_step += 1
            actions = {}
            # print("timestep : ", self.step)
            # print("Start calculate action ....")
            for agent_id, obs in obss.items():
                policy = self.policies[agent_id]
                action, new_state, _ = policy.compute_single_action(obs, state=rnn_states[agent_id],
                                                                    info=infos[agent_id],
                                                                    explore=False)
                actions[agent_id] = action
                # print(agent_id, " :", policy.__class__, " : ", action)
            next_obss, rewards, dones, infos = self.env.step(actions)

            for agent_id, reward in rewards.items():
                rewards_all[agent_id].append(reward)
                episode_reward[agent_id] += reward

            step_balances = {}
            for agent_id in rewards.keys():
                step_balances[agent_id] = self.env.world.facilities[
                    Utils.agentid_to_fid(agent_id)].economy.step_balance.total()
            # print(env.world.economy.global_balance().total(), step_balances, rewards)
            tracker.add_sample(0, episode_step-1, self.env.world.economy.global_balance().total(), step_balances, rewards)

            done = any(dones.values())

            if done:
                obss = self.env.reset(eval=True)
                episode_steps.append(episode_step)
                episode_step = 0
                for agent_id, reward in episode_reward.items():
                    episode_reward_all[agent_id].append(reward)
                    episode_reward[agent_id] = 0
                break
            else:
                obss = next_obss
        infos = {
            "rewards_all": rewards_all,
            "episode_reward_all": episode_reward_all,
            "epsilon": self.policies[self.policies_to_train[0]].epsilon,
            "all_step": self.step,
            "episode_step": sum(episode_steps) / len(episode_steps),
            "profit": tracker.get_retailer_profit(),
        }
        return infos

    def load_data(self, eval=False):
        # self.switch_mode(eval=False)
        print(f"  == start load data eval={eval} == ")

        obss = self.env.reset(eval=eval)
        _, infos = self.env.state_calculator.world_to_state(self.env.world)
        episode_step = 0

        uncontrollable_part_state = {key: infos[key][self.uncontrollable_part_state_key].copy() for key in self.policies_to_train}
        uncontrollable_part_pred = {key: infos[key]['uncontrollable_part_pred'].copy() for key in self.policies_to_train}


        for i in range(self.training_steps):
            episode_step += 1
            actions = {}
            for agent_id, obs in obss.items():
                policy = self.policies[agent_id]
                action, new_state, _ = policy.compute_single_action(obs, state=None,
                                                                    info=infos[agent_id],
                                                                    explore=True)
                actions[agent_id] = action
            next_obss, rewards, dones, infos = self.env.step(actions)

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
                                                         infos[agent_id][self.uncontrollable_part_state_key],
                                                         agent_id, eval=eval)
                uncontrollable_part_state[agent_id] = infos[agent_id][self.uncontrollable_part_state_key].copy()
                uncontrollable_part_pred[agent_id] = infos[agent_id]['uncontrollable_part_pred'].copy()

            if done:
                break
            else:
                obss = next_obss
        print(f"  == load data end episode len={episode_step}==")

