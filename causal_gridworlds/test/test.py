from windy_gridworlds import WindyGridworldEnv
env = WindyGridworldEnv()
env.reset()
done = False
action = 0

while not done:
    action = env.action_space.sample()
    _ = env.step(action)
    ob = _[0]
    r = _[1]
    done = _[2]
    print("Hey")
    a =1


