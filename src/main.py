# Need a replay buffer class
# Need a class for a target Q network (function of state, action)
# We will use batch normalization
# The policy is deterministic, how to handle explore exploit?
# Deterministic policy means outputs the actual action insteat of a probability distribution
# Will need a way to bound the actions to the environment limits
# We gave two actor abd two critic networks, a target for each
# Updates are soft, according to theta_prime = tau * theta + (1 - tau) * theta_prime, with tau << 1
# The targert actor is just the evaluation actor plus some noise process
# They used Ornstein-Uhlenbeck noise for exploration

