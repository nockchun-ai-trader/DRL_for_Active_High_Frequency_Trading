import multiprocessing

multiprocessing.set_start_method('fork', force=True)

import yaml
from pathos.multiprocessing import ProcessPool
from stable_baselines3 import PPO

from data_manager import *
from environment import StockEnv
# from stable_baselines3 import PPO

def run_single_env(j, day, args, model_path):
    """
    Instead of receiving 'model' as a PPO object, receive a path (string).
    Load the model inside the worker, so it doesn't need to be pickled.
    """

    # Load the model inside the worker
    model = PPO.load(model_path, device='cpu')


    data = Data(args)
    file_name, test_files = data.load_test_file(day)
    files_in_dir = os.listdir(os.path.join(os.getcwd(), args.data_dir, 'test_data'))
    test_csv_files = [f for f in files_in_dir if f.endswith('.csv')]
    n_files = len(test_csv_files)

    # n_files = len(os.listdir(os.path.join(os.getcwd(), args.data_dir, 'test_data')))

    env = StockEnv(test_files[0], test_files[1], True, args)
    state = env.reset()
    # print('Env reset to state:', state)

    env_steps = env_reward = env_pos = 0
    profit_per_trade = []

    while True:
        env_steps += 1

        action, _ = model.predict(state)
        state, reward, done, obs = env.step(action)
        env_pos += obs['closed']
        env_reward += reward
        if obs['closed']:
            profit_per_trade += [[reward, obs['open_pos'], obs['closed_pos'], obs['position'], obs['action']]]

        if done:
            break

    return [[file_name, j, env_steps, env_pos, profit_per_trade, env_reward]], n_files * j + day

def eval_agent(args, save_directory):
    save_dir = 'runs_results/' + save_directory
    os.makedirs(save_dir + '/', exist_ok=True)

    # Instead of loading the model here and passing it down,
    # just store the path for each worker to load.
    model_path = os.path.join('runs', save_directory, 'agent')

# def eval_agent(args, save_directory):
#     save_dir = 'runs_results/' + save_directory
#     os.makedirs(save_dir + '/', exist_ok=True)

    # model = PPO.load(os.path.join('runs/' + save_directory, 'agent'), device='cpu')

    with open(os.path.join(save_dir, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)
    files_in_dir = os.listdir(os.path.join(os.getcwd(), args.data_dir, 'test_data'))
    test_csv_files = [f for f in files_in_dir if f.endswith('.csv')]
    n_test_files = len(test_csv_files)
    jobs_to_run = n_test_files * args.eval_runs_per_env
    pool = ProcessPool(multiprocessing.cpu_count())

    for ret, n in pool.uimap(
        run_single_env, 
        np.reshape([[i] * n_test_files for i in range(args.eval_runs_per_env)],jobs_to_run), [*range(n_test_files)] * args.eval_runs_per_env,
        [args] * jobs_to_run, 
        [model_path] * jobs_to_run
        ):
        np.save(save_dir + f'/eval{n}', np.array(ret, dtype=object), allow_pickle=True)
