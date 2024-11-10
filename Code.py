import numpy as np

# Parameters
T = 10          # Time horizon
M = 5           # Max number of calls in the system
lambda1 = 0.5   # Arrival rate of type 1 customers
lambda2 = 0.3   # Arrival rate of type 2 customers
mu1 = 1         # Service rate of representative 1
mu2 = 2         # Service rate of representative 2
dt = 0.1        # Time step for dynamic programming approximation
learning_rate = 0.01  # Learning rate for PPO
gamma = 0.99    # Discount factor for future rewards

class CallCenterOptimizer:
    def __init__(self, T, M, lambda1, lambda2, mu1, mu2, dt):
        self.T = T
        self.M = M
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu1 = mu1
        self.mu2 = mu2
        self.dt = dt
        self.gamma = gamma
        # Initialize value function table with high costs
        self.V = np.full((int(T/dt) + 1, M + 1, M + 1, 2), np.inf)
        self.V[:, 0, 0, :] = 0  # Zero cost for empty queues

    def expected_cost(self, q1, q2, assign_to_rep1, customer_type):
        # Calculate cost based on queue length and service rates
        if assign_to_rep1:
            return (q1 + 1) / self.mu1 + q2 / self.mu2 if customer_type == 0 else q1 / self.mu1 + (q2 + 1) / self.mu2
        else:
            return q1 / self.mu1 + (q2 + 1) / self.mu2

    def dynamic_programming_step(self):
        # Populate the value function table using dynamic programming
        for t in range(1, int(self.T / self.dt) + 1):
            for q1 in range(self.M + 1):
                for q2 in range(self.M + 1):
                    for customer_type in range(2):
                        if q1 + q2 > self.M:
                            continue
                        # Compute costs for assignments to representatives
                        cost_rep1 = self.expected_cost(q1, q2, True, customer_type)
                        cost_rep2 = self.expected_cost(q1, q2, False, customer_type)

                        # Update value function for assignment to rep1
                        if q1 + 1 <= self.M:
                            self.V[t, q1 + 1, q2, customer_type] = min(self.V[t, q1 + 1, q2, customer_type],
                                                                        cost_rep1 + self.V[t - 1, q1, q2, customer_type])

                        # Update value function for assignment to rep2
                        if q2 + 1 <= self.M:
                            self.V[t, q1, q2 + 1, customer_type] = min(self.V[t, q1, q2 + 1, customer_type],
                                                                        cost_rep2 + self.V[t - 1, q1, q2, customer_type])

    def optimal_assignment(self, t, q1, q2, customer_type):
        # Decide the optimal assignment based on the minimal cost
        if t < 0 or q1 + q2 > self.M:
            return None
        elif customer_type == 0 and self.V[t, q1 + 1, q2, 0] <= self.V[t, q1, q2 + 1, 0]:
            return "Assign to Rep 1"
        elif customer_type == 0 and self.V[t, q1 + 1, q2, 0] > self.V[t, q1, q2 + 1, 0]:
            return "Assign to Rep 2"
        elif customer_type == 1:
            return "Assign to Rep 2"

    def approximate_value_function(self, q1, q2):
        # Placeholder for a simple linear approximation of the value function
        # For example, V_approx = alpha * q1 + beta * q2
        alpha, beta = 0.5, 0.3  # Example coefficients
        return alpha * q1 + beta * q2

    def train_policy_with_PPO(self, episodes=1000):
        for episode in range(episodes):
            pass

    def evaluate_policy(self):
        pass


# Run an example with the optimizer
optimizer = CallCenterOptimizer(T, M, lambda1, lambda2, mu1, mu2, dt)
optimizer.dynamic_programming_step()

# Test an example state for optimal assignment
t = 50  # Example time step
q1 = 3
q2 = 1
customer_type = 0
optimal_action = optimizer.optimal_assignment(t, q1, q2, customer_type)
print(f"At time {T - t * dt} with state ({q1}, {q2}, customer type = {customer_type}): Optimal action is to {optimal_action}")
