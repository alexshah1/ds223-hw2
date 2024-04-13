from abc import ABC, abstractmethod
import logging
from logs import CustomFormatter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

logging.basicConfig
logger = logging.getLogger("MAB Application")

# Create console handler with a higher log level
ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

# logger.debug("debug message")
# logger.info("info message")
# logger.warning("warning message")
# logger.error("error message")
# logger.critical("critical message")


class Bandit(ABC):
    """
    Abstract base class for bandit algorithms.

    This class defines the common interface for all bandit algorithms.
    Subclasses must implement the abstract methods to provide specific
    functionality for each algorithm.

    Methods:
        __init__(self, p): Initializes a new instance of the Bandit class.
        __repr__(self): Returns a string representation of the Bandit object.
        pull(self): Simulates pulling the bandit arm and returns the outcome.
        update(self): Updates the internal state of the bandit algorithm.
        experiment(): Runs an experiment using the bandit algorithm.
        report(): Generates a report with average reward and regret.
    """

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @classmethod
    @abstractmethod
    def experiment():
        pass

    @classmethod
    @abstractmethod
    def report():
        # store data in csv
        # log average reward (use f strings to make it informative)
        # log average regret (use f strings to make it informative)
        pass

#--------------------------------------#



class Visualization():
    """
    A class that provides methods for visualizing the performance of bandit algorithms.
    """

    @classmethod
    def plot1(cls, rewards, num_trials, optimal_bandit_reward):
        """
        Plots the average reward convergence of bandit algorithms.

        :param rewards: A list of rewards obtained in each trial.
        :param num_trials: The total number of trials.
        :param optimal_bandit_reward: The reward of the optimal bandit.
        """
        # Visualize the performance of each bandit: linear and log
        # Average rewards
        cumulative_rewards = np.cumsum(rewards)
        average_reward = cumulative_rewards / (np.arange(num_trials) + 1)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(average_reward, label="Average Reward")
        ax[0].axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        ax[0].legend()
        ax[0].set_title("(Linear Scale)")
        ax[0].set_xlabel("Number of Trials")
        ax[0].set_ylabel("Average Reward")

        ax[1].plot(average_reward, label="Average Reward")
        ax[1].axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        ax[1].legend()
        ax[1].set_title("(Log Scale)")
        ax[1].set_xlabel("Number of Trials")
        ax[1].set_ylabel("Average Reward")
        ax[1].set_yscale("log")
        
        fig.suptitle("Average Reward Convergence")

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot2(cls, rewards, num_trials, optimal_bandit_reward):      
        """
        Plots the average regret convergence of bandit algorithms.

        :param rewards: A list of rewards obtained in each trial.
        :param num_trials: The total number of trials.
        :param optimal_bandit_reward: The reward of the optimal bandit.
        """
        # Visualize the performance of each bandit: linear and log
        # Average regrets
        cumulative_rewards = np.cumsum(rewards)
        # Cumulative regret is the difference between the reward of the optimal bandit and the cumulative reward
        cumulative_regrets = optimal_bandit_reward * np.arange(num_trials) - cumulative_rewards
        average_regrets = cumulative_regrets / (np.arange(num_trials) + 1)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(average_regrets, label="Average Regret")
        ax[0].legend()
        ax[0].set_title("(Linear Scale)")
        ax[0].set_xlabel("Number of Trials")
        ax[0].set_ylabel("Average Regret (Reward lost)")

        ax[1].plot(average_regrets, label="Average Regret")
        ax[1].legend()
        ax[1].set_title("(Log Scale)")
        ax[1].set_xlabel("Number of Trials")
        ax[1].set_ylabel("Average Regret (Reward lost)")
        ax[1].set_yscale("log")
        
        fig.suptitle("Average Regret Convergence")

        plt.tight_layout()
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """An implementation of the epsilon-greedy algorithm for the multi-armed bandit problem."""
    def __init__(self, m: float):
        # Reward for the bandit
        self.m = m
        
        # Estimated reward for the bandit
        self.m_estimate = 0
        
        # Number of times the bandit has been pulled
        self.N = 0
        
    def pull(self):
        """
        Pull the bandit arm and return a random number from a normal distribution with mean m.

        :return:float: A random number from a normal distribution with mean m.
        """
        return np.random.randn() + self.m
        
    def update(self, x):
        """
        Update the bandit's estimated reward and counter based on the received reward.

        :param x: The reward received from the bandit.
        """
        # Increment the counter
        self.N += 1
        
        # Update the estimated reward for the bandit
        self.m_estimate = ((self.N - 1)*self.m_estimate + x) / self.N
        
    def __repr__(self):
        """Return a string representation of the bandit."""
        return f"An Arm with {self.m} Reward"
    
    @classmethod
    def experiment(cls, bandit_probabilities, num_trials, initial_epsilon=0.1, min_epsilon=0.02):
        """
        Run an experiment to evaluate the performance of the bandit algorithm.

        :param bandit_probabilities: A list of probabilities representing the true success probabilities of each bandit.
        :param num_trials: The number of trials to run the experiment.
        :param initial_epsilon: The initial value of epsilon for the epsilon-greedy algorithm. (Default value = 0.1)
        :param min_epsilon: The minimum value of epsilon for the epsilon-greedy algorithm. (Default value = 0.02)
        :return: A tuple containing the bandits and the rewards obtained during the experiment.
        """
        logger.info("Initializing the Epsilon Greedy algorithm experiment.")
        
        # Initialize the bandits
        bandits = [EpsilonGreedy(p) for p in bandit_probabilities]

        # Initialize the rewards and counters
        rewards = []
        num_times_explored = 0
        num_times_exploited = 0
        num_times_chosen_optimal = 0

        # Pick the optimal bandit
        optimal_bandit = np.argmax([b.m for b in bandits])
        logger.info(f"Optimal bandit: {optimal_bandit}")

        for i in range(num_trials):
            # Decaying epsilon: start at initial_epsilon and decay up to min_epsilon (change epsilon every 100 trials)
            if np.random.random() < max(initial_epsilon / (i // 100 + 1), min_epsilon):
                # Explore
                num_times_explored += 1
                chosen_bandit = np.random.randint(len(bandits))
            else:
                # Exploit
                num_times_exploited += 1
                chosen_bandit = np.argmax([b.m_estimate for b in bandits])

            # Optimal bandit chosen, increment counter
            if chosen_bandit == optimal_bandit:
                num_times_chosen_optimal += 1

            # Pull the arm for the chosen bandit
            x = bandits[chosen_bandit].pull()

            # Update the rewards log
            rewards.append(x)

            # Update the reward estimate for the chosen bandit
            bandits[chosen_bandit].update(x)

        return bandits, rewards
    
    @classmethod
    def report(cls, bandit_rewards, num_trials):
        """
        Runs an experiment using the Epsilon Greedy algorithm to estimate the mean rewards of bandits.
        Saves the mean rewards per bandit to a CSV file.
        Prints the estimated mean and true mean for each bandit.
        Plots the rewards and regret over the number of trials.
        Calculates and prints the total reward and total regret.

        :param bandit_rewards: A list of rewards for each bandit.
        :param num_trials: The number of trials to run the experiment.
        """
        logger.info("Initializing the Epsilon Greedy algorithm report.")
        
        # Run an experiment
        bandits, rewards= EpsilonGreedy.experiment(bandit_rewards, num_trials)
        
        # Save mean rewards per bandit to a CSV file
        with open("epsilon_greedy.csv", "w") as f:
            f.write(f"bandit_id,true_mean,mean_estimate,times_pulled,algorithm\n")
            for i, b in enumerate(bandits):
                f.write(f"{i},{b.m},{b.m_estimate},{b.N},Epsilon Greedy\n")
                
        # Print the estimates for the bandits
        for i, b in enumerate(bandits):
            logger.info(f"Bandit {i} - Estimated Mean: {b.m_estimate:.4f}, True Mean: {b.m:.4f}")
        
        optimal_bandit_reward = max(bandit_rewards)
        
        Visualization.plot1(rewards, num_trials, optimal_bandit_reward)
        Visualization.plot2(rewards, num_trials, optimal_bandit_reward)
        
        sum_reward = sum(rewards)
        sum_regret = optimal_bandit_reward * num_trials - sum_reward
        
        logger.info(f"Total Reward: {sum_reward:.4f}; Average Reward: {sum_reward/num_trials:.4f}")
        logger.info(f"Total Regret: {sum_regret:.4f}: Average Regret: {sum_regret/num_trials:.4f}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    """An implementation of the Thompson Sampling algorithm for the multi-armed bandit problem."""
    def __init__(self, m):
        # Store the true mean for sampling
        self.m = m
        
        # Parameters for mu - prior is N(0,1)
        self.m_estimate = 0
        self.lambda_ = 1
        
        # Precision
        self.tau = 1
        self.sum_x = 0
    
        # Number of times the bandit has been pulled
        self.N = 0
        
    def pull(self):
        """
        Pull the bandit arm and return a random number from a normal distribution with mean m.
        
        :return:float: A random number from a normal distribution with mean m.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.m
    
    def sample(self):
        """
        Pull the bandit arm and return a random number from a normal distribution with the estimated mean.
        
        :return:float: A random number from a normal distribution with the estimated mean.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m_estimate
    
    def update(self, x):
        """
        Update the bandit's distribution parameters based on the observed reward.

        :param x: The observed reward.
        """
        # Increment counter
        self.N += 1
        
        # Update the distribution parameters
        self.lambda_ += self.tau
        self.sum_x += x
        self.m_estimate = (self.tau * self.sum_x)/self.lambda_
        
    def __repr__(self):
        """Return a string representation of the bandit."""
        return f"An Arm with {self.m} Reward"
        
    @classmethod
    def experiment(cls, bandit_rewards, num_trials, plot=False):
        """
        Run an experiment using the Thompson Sampling algorithm.

        :param bandit_rewards: A list of true means for each bandit arm.
        :param num_trials: The number of trials to run the experiment.
        :param plot: Whether to plot the bandit distributions during the experiment. (Default value = False)
        :return: The bandits and the rewards obtained during the experiment.
        """
        logger.info("Initializing the Thompson Sampling algorithm experiment.")

        bandits = [ThompsonSampling(p) for p in bandit_rewards]
        
        # Take 6 sample points from 10 to num_trials
        sample_points = list(np.linspace(1, num_trials, 5, dtype=int))
        sample_points[0] = 10
        sample_points[-1] -= 1
        sample_points.insert(1, 500)
        
        rewards = []
        res = []
        for i in range(num_trials):
            # Choose the bandit with the highest sampled value
            j = np.argmax([b.sample() for b in bandits])
            
            if plot and i in sample_points:
                # To later plot the PDFs of the bandits and the true means (this will help us understand how close we got to the true means, and the highest peak will be the optimal bandit)
                res.append([(b.m_estimate, b.m, b.lambda_, b.N) for b in bandits])
                
                
            # Pull the chosen bandit
            x = bandits[j].pull()
            
            # Add the reward to the list of rewards
            rewards.append(x)
            
            # Update the chosen bandit
            bandits[j].update(x)
        
        if plot:     
            ThompsonSampling.plot(sample_points, res)
        
        return bandits, rewards
    
    @classmethod
    def report(cls, bandit_rewards, num_trials, plot=False):
        """
        Run an experiment using the Thompson Sampling algorithm and report the results.

        :param bandit_rewards: A list of true means for each bandit arm.
        :param num_trials: The number of trials to run the experiment.
        """
        logger.info("Initializing the Thompson Sampling algorithm report.")

        # Run an experiment
        bandits, rewards = ThompsonSampling.experiment(bandit_rewards, num_trials, plot=plot)
        
        # Save mean rewards per bandit to a CSV file
        with open("thompson_sampling.csv", "w") as f:
            f.write(f"bandit_id,true_mean,mean_estimate,times_pulled,algorithm\n")
            for i, b in enumerate(bandits):
                f.write(f"{i},{b.m},{b.m_estimate},{b.N},Thompson Sampling\n")
        logging.info("Saved the results to thompson_sampling.csv.")
        
        # Print the estimates for the bandits
        for i, b in enumerate(bandits):
            logger.info(f"Bandit {i} - Estimated Mean: {b.m_estimate:.4f}, True Mean: {b.m:.4f}")
        
        optimal_bandit_reward = max(bandit_rewards)
        
        Visualization.plot1(rewards, num_trials, optimal_bandit_reward)
        Visualization.plot2(rewards, num_trials, optimal_bandit_reward)
           
        sum_reward = sum(rewards)
        sum_regret = optimal_bandit_reward * num_trials - sum_reward
        
        logger.info(f"Total Reward: {sum_reward:.4f}; Average Reward: {sum_reward/num_trials:.4f}")
        logger.info(f"Total Regret: {sum_regret:.4f}: Average Regret: {sum_regret/num_trials:.4f}")
        
    @classmethod
    def plot(self, sample_points, res):
        """
        Plot the bandit distributions for several trials.

        :param sample_points: The list of sample points.
        :param res: The list of tuples containing trial results.
                    Each tuple should contain the following elements:
                    - m_estimate: The estimated mean.
                    - m: The true mean.
                    - lambda_: The precision parameter.
                    - N: The number of trials.
        """
        x = np.linspace(-3, 6, 200)
        
        rows = 2
        cols = len(sample_points)//2
        fig, ax = plt.subplots(rows, cols, figsize=(15, 5))
        for i, (trial, vals) in enumerate(zip(sample_points, res)):
            for (m_estimate, m, lambda_, N) in vals:
                y = norm.pdf(x, m_estimate, np.sqrt(1. / lambda_))
                p = ax[i//cols][i%cols].plot(x, y, label=f"est_m: {m_estimate:.2f}, #tr: {N}")
                ax[i//cols][i%cols].axvline(x=m, linestyle="--", c=p[0].get_color())
                ax[i//cols][i%cols].set_title(f"{trial} trials")
                ax[i//cols][i%cols].legend()
        
        fig.suptitle("Bandit distributions after different number of trials")
        plt.show()

def comparison(bandit_rewards, num_trials, different_plots=False):
    """
    Compare the performance of Epsilon Greedy and Thompson Sampling algorithms
    by visualizing the average reward convergence over a given number of trials.

    :param bandit_rewards: A list of rewards for each bandit.
    :param num_trials: The number of trials to run the algorithms.
    :param different_plots: A boolean indicating whether to plot the performance
                            in side-by-side plots or in one plot. (Default value is False)
    """
    logger.info("Starting the comaprison between Epsilon Greedy and Thompson Sampling algorithms.")

    eg_bandits, eg_rewards = EpsilonGreedy.experiment(bandit_rewards, num_trials)
    ts_bandits, ts_rewards = ThompsonSampling.experiment(bandit_rewards, num_trials)
    
    # Print the estimates for the bandits
    for i, (egb, tsb) in enumerate(zip(eg_bandits, ts_bandits)):
        true_mean = egb.m
        egb_estimate = egb.m_estimate
        egb_diff = abs(egb_estimate-true_mean)
        tsb_estimate = tsb.m_estimate
        tsb_diff = abs(tsb_estimate-true_mean)
        closer = "Thompson Sampling" if egb_diff > tsb_diff else "Epsilon Greedy"
        
        logger.info(f"Bandit {i}:\nEG Est, Mean: {egb_estimate:.4f} ({egb_diff:.2f}), TS Est. Mean: {tsb_estimate:.4f} ({tsb_diff:.2f}), True Mean: {true_mean:.4f} --- {closer} closer {'OPTIMAL' if i == np.argmax(bandit_rewards) else ''}")
    
    optimal_bandit_reward = max(bandit_rewards)
    
    eg_average_rewards = np.cumsum(eg_rewards) / (np.arange(num_trials) + 1)
    ts_average_rewards = np.cumsum(ts_rewards) / (np.arange(num_trials) + 1)
    
    if different_plots:
        # Visualize the performance in side-by-side plots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(eg_average_rewards, label="Average Reward")
        ax[0].axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        ax[0].legend()
        ax[0].set_title("Epsilon Greedy")
        ax[0].set_xlabel("Number of Trials")
        ax[0].set_ylabel("Average Reward")
        ax[0].set_xscale("log")

        ax[1].plot(ts_average_rewards, label="Average Reward")
        ax[1].axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        ax[1].legend()
        ax[1].set_title("Thompson Sampling")
        ax[1].set_xlabel("Number of Trials")
        ax[1].set_ylabel("Average Reward")
        ax[1].set_xscale("log")
        
        fig.suptitle("Average Reward Convergence")

        plt.tight_layout()
        plt.show()
    else:
        # Visualize the performance in one plot
        plt.plot(eg_average_rewards, label="Epsilon Greedy")
        plt.plot(ts_average_rewards, label="Thompson Sampling")
        plt.axhline(optimal_bandit_reward, color="r", linestyle="--", label="Optimal Bandit Reward")
        plt.legend()
        plt.title("Average Reward Convergence")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.xscale("log")
        
        plt.show()

if __name__ == "__main__":
    
    BANDIT_REWARDS = [1, 2, 3, 4]
    NUM_TRIALS = 20000
    
    # Test Epsilon Greedy report
    EpsilonGreedy.report(BANDIT_REWARDS, NUM_TRIALS)
    
    # Test Thompson Sampling report
    ThompsonSampling.report(BANDIT_REWARDS, NUM_TRIALS, plot=True)
    
    # Compare both algorithms
    comparison(BANDIT_REWARDS, NUM_TRIALS, different_plots=False)

# Potential improvements:
# - Make report and experiment methods as class methods, since they're not associated with class instances (already implemented). Alternatively, make them more generic, that is, not tied to a specific class.
# - Do not encompass functions into visualization class, since they don't really have any data to share. Instead, make them standalone functions. Also, use better names for said functions.

# Overall comments:
# - It would have been much easier for the student if the functions had some documentation with minimal requirements.