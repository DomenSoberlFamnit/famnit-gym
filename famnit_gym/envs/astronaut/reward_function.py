def default_reward_function(reward_info):
    if reward_info.get('collision', False):
        return -1

    if reward_info.get('oxygen_picked', False):
        return 1

    return 0