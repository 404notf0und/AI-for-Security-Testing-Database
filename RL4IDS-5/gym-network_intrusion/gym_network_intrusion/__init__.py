from gym.envs.registration import register

register(
    id='network-intrusion-v0',
    entry_point='gym_network_intrusion.envs:NetworkIntrusionEnv',
)

register(
    id='network-intrusion-extrahard-v0',
    entry_point='gym_network_intrusion.envs:NetworkIntrusionExtraHardEnv',
)
