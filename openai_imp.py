# -*- coding: utf-8 -*-
import os
# CPU tarafında bellek ve thread baskısını azalt
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU aramasın, uyarılar azalsın

import pickle
import sys
import time
from argparse import Namespace

import gym
from baselines.a2c.runner import Runner
from baselines.common import set_global_seeds, explained_variance
from baselines.common.cmd_util import make_env
from baselines.common.policies import build_policy
from baselines.ppo2.ppo2 import safemean
from baselines.common.vec_env import DummyVecEnv, VecEnv
from baselines.common.vec_env.test_vec_env import SimpleEnv
# from baselines.run import build_env  # kullanılmıyor

from hx_controller.haxball_gym import Haxball
from hx_controller.haxball_vecenv import HaxballVecEnv  # TEK PROSES env
from hx_controller.openai_model_torneo import A2CModel
from simulator import create_start_conditions
import numpy as np
from collections import deque


if __name__ == '__main__':
    # def build_env():
    #     # gameplay = create_start_conditions()
    #     # env = Haxball(gameplay=gameplay)
    #     env = SimpleEnv(0, (5, ), np.float32)
    #     return env

    # gameplay = create_start_conditions()
    try:
        res = gym.spec('haxball-v0')
    except Exception:
        gym.register(
            id='haxball-v0',
            entry_point='hx_controller.haxball_gym:Haxball',
            kwargs=dict(gameplay=None, max_ticks=2400)
        )

    args_namespace = Namespace(
        alg='a2c',
        env='haxball-v0',
        num_env=None,
        # env='PongNoFrameskip-v4',
        env_type=None, gamestate=None, network=None, num_timesteps=1000000.0, play=False,
        reward_scale=1.0, save_path=None, save_video_interval=0, save_video_length=200, seed=None)

    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None

    from baselines import logger

    def make_vec_env(env_id, env_type, num_env, seed,
                     wrapper_kwargs=None,
                     start_index=0,
                     reward_scale=1.0,
                     flatten_dict_observations=True,
                     gamestate=None):
        """
        Create a wrapped, monitored DummyVecEnv (tek proses için).
        """
        wrapper_kwargs = wrapper_kwargs or {}
        mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        seed = seed + 10000 * mpi_rank if seed is not None else None
        logger_dir = logger.get_dir()

        def make_thunk(rank):
            return lambda: make_env(
                env_id=env_id,
                env_type=env_type,
                mpi_rank=mpi_rank,
                subrank=rank,
                seed=seed,
                reward_scale=reward_scale,
                gamestate=gamestate,
                flatten_dict_observations=flatten_dict_observations,
                wrapper_kwargs=wrapper_kwargs,
                logger_dir=logger_dir
            )

        set_global_seeds(seed)
        return DummyVecEnv([make_thunk(i + start_index) for i in range(num_env)])

    nsteps = 5
    gamma = 0.99
    nenvs = 1
    total_timesteps = int(10e7)
    log_interval = 100

    load_path = 'models2/ciao.h5'
    # model_i = 8
    model_i = ''

    play = bool(int(os.getenv('PLAY', False)))
    if play:
        nenvs = 2

    # env = make_vec_env(env_id='haxball-v0', env_type=None, num_env=nenvs, seed=None)
    # env = HaxballVecEnv(num_fields=nenvs, max_ticks=2400*2)
    max_ticks = int(60 * 2 * (1 / 0.1))
    if play:
        max_ticks = int(60 * 1 * (1 / 0.1))

    # *** KRİTİK: SubProc yerine tek-proses env ***
    env = HaxballVecEnv(num_fields=nenvs, max_ticks=max_ticks)

    # policy = build_policy(env=env, policy_network='lstm', num_layers=4, num_hidden=128)
    policy = build_policy(env=env, policy_network='mlp', num_layers=4, num_hidden=256)

    # Model
    model = A2CModel(
        policy,
        model_name=('p' + str(model_i)) if isinstance(model_i, int) and model_i > 0 else 'a2c_model',
        env=env,
        nsteps=nsteps,
        ent_coef=0.1,
        total_timesteps=total_timesteps,
        lr=7e-4  # 0.005
        # , vf_coef=0.0
    )

    # Kayıt klasörü yoksa oluştur
    try:
        os.makedirs(os.path.dirname(load_path), exist_ok=True)
    except Exception:
        pass

    if load_path is not None and os.path.exists(load_path):
        model.load(load_path)

    if play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = 0
        episode_rew2 = 0
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, rew, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            episode_rew2 += rew[1] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}, \t{}'.format(episode_rew, episode_rew2))
                episode_rew = 0
                episode_rew2 = 0
                obs = env.reset()

    # Runner
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)

    # Batch
    nbatch = nenvs * nsteps

    # Timer
    tstart = time.time()
    last_rewards = []

    graph_names = (
        'policy_entropy', 'value_loss', 'policy_loss', 'values_mean',
        'explained_variance',
        'rewards_mean', 'rewards_min', 'rewards_max', 'rewards_median', 'rewards_std',
        'values_mean', 'values_min', 'values_max', 'values_median', 'values_std'
    )
    graph_data = {k: [] for k in graph_names}

    for update in range(1, total_timesteps // nbatch + 1):
        # Mini-batch
        obs, states, rewards, masks, actions, values, epinfos = runner.run()

        # invert
        inv_obs = env.invert_states(obs)
        obs = np.vstack((obs, inv_obs))
        rewards = np.hstack((rewards, rewards))
        masks = np.hstack((masks, masks))
        inv_actions = env.invert_actions(actions)
        actions = np.hstack((actions, inv_actions))
        values = np.hstack((values, values))

        epinfobuf.extend(epinfos)

        policy_loss, value_loss, policy_entropy = model.train(
            obs, states, rewards, masks, actions, values
        )
        nseconds = time.time() - tstart

        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", str(update))
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular('rewards', np.mean(rewards))
            logger.record_tabular('values', np.mean(values))
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()

            # Grafik verileri
            graph_data['policy_entropy'].append(float(policy_entropy))
            graph_data['value_loss'].append(float(value_loss))
            graph_data['policy_loss'].append(float(policy_loss))
            graph_data['values_mean'].append(np.mean(values))
            graph_data['values_min'].append(np.min(values))
            graph_data['values_max'].append(np.max(values))
            graph_data['values_std'].append(np.std(values))
            graph_data['values_median'].append(np.median(values))
            graph_data['rewards_mean'].append(np.mean(rewards))
            graph_data['rewards_min'].append(np.min(rewards))
            graph_data['rewards_max'].append(np.max(rewards))
            graph_data['rewards_std'].append(np.std(rewards))
            graph_data['rewards_median'].append(np.median(rewards))
            graph_data['explained_variance'].append(float(ev))
            try:
                with open(load_path + '.graph', 'wb') as fp:
                    pickle.dump(graph_data, fp)
            except Exception:
                pass

        if update % 5_000 == 0:
            load_path_i = load_path.replace('.h5', '_' + str(update) + '.h5')
            try:
                model.save(load_path_i)
            except Exception:
                pass
