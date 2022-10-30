from gym.envs.registration import register

register(
    id='vision-aviary-v0',
    entry_point='drone_landing.env:VisionAviary',
)
