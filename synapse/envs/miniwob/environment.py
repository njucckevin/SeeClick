import os

from synapse.envs.miniwob.instance import MiniWoBInstance

MINIWOB_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "html", "miniwob"
)

EXTRA_HTML_TASKS = [
    "click-dialog",
    "click-dialog-2",
    "use-autocomplete",
    "choose-date",
]


class MiniWoBEnv(object):
    def __init__(
        self,
        subdomain: str,
        headless: bool = False,
    ):
        """Creates a new MiniWoBEnv with no instances.
        Must call configure() to set up instances.

        Args:
            subdomain (str): MiniWoB task name (e.g., "click-test")
            headless (bool): Whether to render GUI
        """
        self.subdomain = subdomain
        self.instance = None
        self.headless = headless
        self.task = None

    def configure(self, seed: int = None, **kwargs):
        """Creates the required number of MiniWoBInstance.

        Args:
            seed (int): Random seed to set the instance;

        kwargs are passed into the constructor of MiniWoBInstance:
            headless (bool): Whether to render GUI
            base_url (str): Base URL, which is usually one of the following
                - http://localhost:8000/     (served by http-serve)
                - file:///path/to/miniwob-plusplus/html/
            cache_state (bool): Whether to cache and return the initial
                state; only make sense if the task interface never changes
            threading (bool): Whether to run the instances in separate threads
            reward_processor (callable; optional): A function that takes
                the metadata and return a reward (see miniwob.reward)
            seeds (list[object]): Random seeds to set for each instance;
                len(seeds) must be equal to num_instances.
            wait_ms (float): Pause the instance after each action for this
                amount of time (in milliseconds).
            block_on_reset (bool): On reset, block until the page loads.
            refresh_freq (int): Every this number of episodes,
                refresh the page at the beginning of the next episode.
                Takes time but cleans up any lingering states and memory leaks.
                *** Must specify `seeds` at each reset call.
            initial_mode (str): Initial data mode (e.g., "train", "test")
        """
        assert seed is not None, "seed must be specified"
        if self.instance is not None:
            self.instance.close()
        self.instance = None
        self.instance = MiniWoBInstance(
            index=0,
            subdomain=self.subdomain,
            seed=seed,
            headless=self.headless,
            base_url=f"file://{MINIWOB_DIR}",
            wait_ms=1000.0,
            refresh_freq=1,
            **kwargs,
        )
        self.instance.start()
        self.instance.wait()

    def reset(
        self,
        seed: int = None,
        record_screenshots: bool = False,
    ) -> str:
        """Forces stop and start all instances.

        Args:
            seed (int): Random seed to set the instance
            record_screenshots (bool): Whether to record screenshots of the states.
        Returns:
            obs (str)
        """
        self.configure(seed=seed)
        self.set_record_screenshots(record_screenshots)
        states = [None]
        self.instance.call(self.instance.reset, states, seed)
        self.instance.wait()
        obs = self.state2html(states)
        self.task = states[0].utterance

        return obs

    def step(self, action):
        """Applies an action on each instance and returns the results.

        Args:
            action (MiniWoBAction)

        Returns:
            state (MiniWoBState)
            reward (float)
            done (bool)
            info (dict)
        """
        states = [None]
        rewards = [-1.0]
        dones = [True]
        info = {"n": [{}]}
        self.instance.call(
            self.instance.step, action, states, rewards, dones, info["n"]
        )
        self.instance.wait()
        obs = self.state2html(states)

        return obs, rewards[0], dones[0], info["n"][0]

    def set_record_screenshots(self, record_screenshots):
        """Adjust whether the record the screenshots of the states.

        Args:
            record_screenshots (bool)
        """
        self.instance.record_screenshots = record_screenshots

    def close(self):
        self.instance.call(self.instance.close)
        self.instance.wait()

    def get_task(self):
        return self.task

    def state2html(self, states: list) -> str:
        if states[0] is not None:
            obs = states[0].html_body
            if self.subdomain in EXTRA_HTML_TASKS:
                obs += states[0].html_extra
        else:
            obs = None

        return obs
