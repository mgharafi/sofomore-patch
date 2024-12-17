import comocma
from comocma import Sofomore, SofomoreDataLogger
from moarchiving import BiobjectiveNondominatedSortedList as NDA

import os
from pathlib import Path
from abc import ABC, abstractmethod, update_abstractmethods

from zmq import has


def kernelBuilder(Algorithm, other_attributes={}):

    class Solver(
        KernelInterface,
        Algorithm,
    ):
        def __new__(cls, *args, **kwargs):

            alg_cls = cls.__mro__[3]

            for k in cls.__abstractmethods__:
                if k in alg_cls.__dict__:
                    setattr(cls, k, alg_cls.__dict__[k])

            for k, v in other_attributes.items():
                setattr(cls, k, v)

            update_abstractmethods(cls)

            return super().__new__(cls)

    return Solver


class KernelInterface(ABC):
    def __init__(self, *args, **kwargs):
        if "reference_point" in kwargs:
            self.reference_point = kwargs["reference_point"]
            del kwargs["reference_point"]

        super(type(self).__mro__[1], self).__init__(*args, **kwargs)
        self.objective_values = None
        self._last_offspring_f_values = None

    def copy(self, defaults=None):
        # To implement for specific single objective algorithms
        new_kernel = type(self)(**defaults)
        new_kernel.objective_values = self.objective_values
        new_kernel._last_offspring_f_values = self._last_offspring_f_values

        return new_kernel

    @abstractmethod
    def tell(self):
        pass

    @abstractmethod
    def ask(self):
        pass

    @property
    @abstractmethod
    def incumbent(self):
        pass


class IndicatorFront(comocma.IndicatorFront):
    def set_kernel(self, kernel, moes, lazy=True):
        """Set empirical front for evolving the given kernel.

        By default, make changes only when kernel has changed.

        Details: ``moes.reference_point`` and, in case, its attribute
        with name `self.list_attribute: str` is used.
        """
        if lazy and kernel == self.kernel:
            return

        if not hasattr(kernel, "reference_point"):
            kernel.reference_point = moes.reference_point

        if kernel.list_attribute:
            list_of_pairs = getattr(kernel, kernel.list_attribute)
            if isinstance(list_of_pairs[0], type(kernel)):
                list_of_pairs = [
                    k.objective_values for k in list_of_pairs if k != kernel and k.objective_values is not None
                ]

            # print(list_of_pairs)
            self.front = self.NDA(list_of_pairs, kernel.reference_point)
        else:
            self.front = self.NDA(
                [k.objective_values for k in moes if k != kernel],
                kernel.reference_point,
            )
        self.kernel = kernel


class Multistart:
    def __init__(self, kernel_class: KernelInterface, sofomore_options: dict = {}):
        self.sofomore_options = sofomore_options
        self.kernel_class = kernel_class
        self.moes = None
        self.starts = []

        if 'list_attribute' in self.sofomore_options:
            self.list_attribute = self.sofomore_options['list_attribute']
        else:
            self.list_attribute = "group"

    def add_start(self, opts: list):

        start_solvers = []
        list_attribute = self.list_attribute

        for opt in opts:
            solver = self.kernel_class(**opt)
            start_solvers.append(solver)
            solver.list_attribute = list_attribute
            setattr(solver, list_attribute, start_solvers)

        if self.moes is None:
            self.moes = SofomorePatch(start_solvers[:], **self.sofomore_options)

        else:
            self.moes.add(start_solvers[:])
            self.moes._told_indices = range(len(self.moes))

        self.starts.append(start_solvers[:])

        return start_solvers[:]

    def build_run(self):
        return self.moes
        # return SofomorePatch(self.solvers, **self.sofomore_options)


class RestartSquencer:
    def __init__(
        self,
        restarts,
        Kernel: KernelInterface,
        budget: int,
        sofomore_options: dict = {},
    ):
        self.restarts = iter(restarts)
        self.run_manager = Multistart(Kernel, sofomore_options)
        self.current_index = 0
        self.restarts_kernels = []
        self.budget = budget

    def __iter__(self):
        return self

    def __next__(self):
        restart_options = next(self.restarts)
        self.current_index += 1
        reference_points = self.new_reference_points()

        budget = self.compute_budget(reference_points, restart_options[:])
        current_budget = self.budget
        popsize = budget if budget is not None else None
        added_solvers = []

        for ref in reference_points:

            ref_kernels = []
            new_options = {"reference_point": ref}


            if hasattr(self, "archive"):
                x0 = self.select_incumbent(ref, self.archive)
                if x0 is not None:
                    new_options["x0"] = x0

            for ropt in restart_options:
                for opt in ropt:
                    if popsize is not None:
                        current_budget -= (popsize + 1)
                        if current_budget < (popsize + 1):
                            popsize += current_budget
                    opt.update(new_options)
                    opt['inopts']['popsize'] = popsize
                ref_kernels.append(self.run_manager.add_start(ropt))
            added_solvers.append(ref_kernels)
        self.restarts_kernels.append(added_solvers[:])

        return self.run_manager.build_run()
    
    def compute_budget(self, reference_points, restart_options):
        budget = self.budget
        number_of_restart_kernels = sum([len(start) for start in restart_options])
        number_of_new_kernels = len(reference_points) * number_of_restart_kernels

        # print(f"Number of new kernels: {number_of_new_kernels}")

        if budget is not None:
            if number_of_new_kernels > budget:
                raise ValueError(
                    f"The number of new kernels ({number_of_new_kernels}) is larger than the budget ({budget})."
                )
            
            popsize = (budget - number_of_new_kernels) // number_of_new_kernels

            return popsize

        else:
            return None

    def select_incumbent(self, reference_point, bidict_archive):
        if hasattr(self.run_manager, "moes"):
            moes = self.run_manager.moes
            if hasattr(moes, "archive"):
                nd_archive = NDA(moes.archive, reference_point)
                if len(nd_archive)>0:
                    max_hv = max(nd_archive, key=lambda x: NDA([x], reference_point).hypervolume)
                    # local_archive = [bidict_archive.inverse[tuple(sol)] for sol in nd_archive]

                    return bidict_archive.inverse[tuple(max_hv)]

        return None



    def new_reference_points(self):
        reference_point = self.run_manager.sofomore_options["reference_point"]
        r1, r2 = reference_point
        refs = [reference_point]

        if self.run_manager.moes is None:
            return refs

        moes = self.run_manager.moes

        # Get the empirical front
        EPF = NDA(
            [k.objective_values for k in moes if k.objective_values is not None],
            reference_point,
        )

        if len(EPF) != 0:
            kinks = [[u[0], v[1]] for u, v in zip(EPF[1:], EPF)]
            refs = [[EPF[0][0], r2], *kinks, [r1, EPF[-1][1]]]

        return refs


class SofomorePatch(Sofomore):
    def __init__(self, *args, patch: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.indicator_front = IndicatorFront()

        if patch:
            self.root = Path(self.logger.name_prefix)
            self.root.mkdir(exist_ok=True)

            self.logger = SofomoreDataLogger(
                self.root.as_posix(), modulo=self.opts["verb_log"]
            )

            self.logger.name_prefix = f"{self.root.as_posix()}{os.sep}"
            self.logger.register(self)

            self.prefix_kernel_loggers()

    def prefix_kernel_loggers(self):
        so_output = Path(self.root) / "so_output"
        so_output.mkdir(exist_ok=True)

        for ikernel, kernel in enumerate(self.kernels):
            assert hasattr(
                kernel, "logger"
            ), "The kernel does not have a logger attribut"
            kernel.logger.name_prefix = (so_output / f"{ikernel}").as_posix()
