def eval_one_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset") # reset policy

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)

