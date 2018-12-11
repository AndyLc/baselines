import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.input import observation_placeholder
from baselines.aux.policies import build_policy


from baselines.aux.utils import Scheduler, find_trainable_variables
from baselines.aux.runner import Runner

from tensorflow import losses

class Model(object):

    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, r0_coef=0.05, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear', head=-1, step_placeholder=None, train_placeholder=None, encoded_x_1=None, encoded_x_2=None):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps
        self.step_placeholder = step_placeholder
        self.train_placeholder = train_placeholder
        self.encoded_x_1 = encoded_x_1
        self.encoded_x_2 = encoded_x_2
        with tf.variable_scope('aux_model' + str(head), reuse=tf.AUTO_REUSE):
            step_model, self.step_placeholder, self.encoded_x_1 = policy(nenvs, 1, sess, observ_placeholder=self.step_placeholder, encoded_x=self.encoded_x_1)
            train_model, self.train_placeholder, self.encoded_x_2 = policy(nbatch, nsteps, sess, observ_placeholder=self.train_placeholder, encoded_x=self.encoded_x_2)

            A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
            ADV = tf.placeholder(tf.float32, [nbatch])
            R = tf.placeholder(tf.float32, [nbatch])
            LR = tf.placeholder(tf.float32, [])

            neglogpac = train_model.pd.neglogp(A)
            entropy = tf.reduce_mean(train_model.pd.entropy())

            pg_loss = tf.reduce_mean(ADV * neglogpac)
            print(train_model.vf)
            print(R)
            vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)
            #r0_loss = losses.mean_squared_error(tf.squeeze(train_model.r), R)

            loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

            params = find_trainable_variables('aux_model' + str(head))
            grads = tf.gradients(loss, params)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            #print("gradiants to update: ", grads)
            trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
            _train = trainer.apply_gradients(grads)

            with tf.name_scope('summaries'):
                a_r = tf.summary.scalar('avg_reward', tf.reduce_mean(R))
                #a_p_l = tf.summary.scalar('avg_pg_loss', tf.reduce_mean(pg_loss))
                #a_v_l = tf.summary.scalar('avg_vf_loss', tf.reduce_mean(vf_loss))
                #a_l = tf.summary.scalar('avg_loss', tf.reduce_mean(loss))
                #merged = tf.summary.merge([a_r, a_p_l, a_v_l, a_l])
                merged = tf.summary.merge([a_r])

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values): #Policy already sampled!! We need to update the critic now.
            advs = rewards - values #For a set of (s, a), we get (r0 - v0, r1 - v1, ...)
            #print("advs: ", advs)
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            #print(td_map)
            #we train out model with observed actions, advs, rewards, cur_lr
            #how is values and rewards calculated though? These cannot be sampled from a single state.
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, summary, _ = sess.run(
                [pg_loss, vf_loss, entropy, merged, _train],
                td_map
            )

            return policy_loss, value_loss, policy_entropy, summary

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)

        self.train_writer = tf.summary.FileWriter('logs/aux/' + str(head),
                                              sess.graph)

        tf.global_variables_initializer().run(session=sess)


def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    r0_coef=0.05,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    heads=2,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''
    set_global_seeds(seed)

    nenvs = env.num_envs
    print("network type: ", network)

    ob_space = env.observation_space
    policies = [build_policy(env, network, i, **network_kwargs) for i in range(heads)]

    #TODO: connect the nets from observations to some shared network....

    models = []
    model = Model(policy=policies[0], env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, r0_coef=r0_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, head=0)

    models.append(model)

    splaceh = model.step_placeholder
    tplaceh = model.train_placeholder
    encoded_x_1 = model.encoded_x_1
    encoded_x_2 = model.encoded_x_2

    for i in range(1, heads):
        model = Model(policy=policies[i], env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, r0_coef=r0_coef,
            max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, head=i, step_placeholder=splaceh, train_placeholder=tplaceh, encoded_x_1=encoded_x_1, encoded_x_2=encoded_x_2)
        models.append(model)

    if load_path is not None:
        model.load(load_path)

    #TODO: create multiple environments, each with a different length cartpole.
    runners = []
    for _ in range(heads):
        runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
        runners.append(runner)

    nbatch = nenvs*nsteps
    tstart = time.time()
    #print("nbatch: ", nbatch)
    for update in range(0, total_timesteps//nbatch):
        for h in range(heads):
            obs, states, rewards, masks, actions, values = runners[h].run() #We observe these from the environment
            policy_loss, value_loss, policy_entropy, summary = models[h].train(obs, states, rewards, masks, actions, values)
            nseconds = time.time()-tstart
            fps = int((update*nbatch)/nseconds)

            if update % log_interval >= 0 and update % log_interval < heads:
                models[h].train_writer.add_summary(summary, update//log_interval)

                #print(models[h].train_writer)
                #print(policy_loss, value_loss, policy_entropy, summary)

                ev = explained_variance(values, rewards)
                """
                logger.record_tabular("nupdates" + str(update % log_interval), update)
                logger.record_tabular("total_timesteps" + str(update % log_interval), update*nbatch)
                logger.record_tabular("fps" + str(update % log_interval), fps)
                logger.record_tabular("policy_entropy" + str(update % log_interval), float(policy_entropy))
                logger.record_tabular("value_loss" + str(update % log_interval), float(value_loss))
                logger.record_tabular("explained_variance" + str(update % log_interval), float(ev))
                logger.dump_tabular()
                """

    return model
