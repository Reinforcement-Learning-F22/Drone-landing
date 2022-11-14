from gym.envs.registration import register

register(
    id='landing-aviary-v0',
    entry_point='drone_landing.env:LandingAviary',
)
register(
    id='hover-aviary-v1',
    entry_point='drone_landing.env:HoverAviary',
)
register(
    id='alignment-aviary-v0',
    entry_point='drone_landing.env:AlignmentAviary',
)

