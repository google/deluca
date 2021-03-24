from gym.envs.registration import register
register(
    id='Fixswimmer-v1',
    entry_point='deluca.envs.mbbl.gym_env.fix_swimmer.fixed_swimmer:fixedSwimmerEnv'
)
