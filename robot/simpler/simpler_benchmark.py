from experiments.robot.simpler.simpler_utils import get_simpler_env

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


###

task_map = {
    "simpler_widowx": [
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket",
    ],
    "simpler_widowx_carrot": [
        "widowx_carrot_on_plate",
    ],
}


class Benchmark:
    def _make_benchmark(self):
        self.tasks = task_map[self.name]

    def get_task(self, i):
        return self.tasks[i]

    def make(self, *args, **kwargs):
        return self.env_fn(*args, **kwargs)

    @property
    def n_tasks(self):
        return len(self.tasks)


class SimplerBenchmark(Benchmark):
    def __init__(self):
        super().__init__()
        self.env_fn = get_simpler_env
        self.state_dim = 7


@register_benchmark
class SIMPLER_WIDOWX(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_widowx"
        self._make_benchmark()


@register_benchmark
class SIMPLER_WIDOWX_CARROT(SimplerBenchmark):
    def __init__(self):
        super().__init__()
        self.name = "simpler_widowx_carrot"
        self._make_benchmark()
