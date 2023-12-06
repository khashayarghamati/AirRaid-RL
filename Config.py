

class Config:
    env_name = 'ALE/AirRaid-v5'
    lr = 1e-4
    exploration_rate = 1
    exploration_rate_decay = 0.99999975
    exploration_rate_min = 0.1
    discount_factor = 0.99

    save_every = 5e5 * 2
    total_episode = 40
