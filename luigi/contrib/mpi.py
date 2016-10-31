"""
MPI Task Scheduling
"""

import logging
import socket
import collections
import multiprocessing
from mpi4py import MPI
from luigi import build, configuration
from luigi.worker import Worker, worker
from luigi.scheduler import Scheduler

log = logging.getLogger('luigi-interface')
COM = MPI.COMM_WORLD


def send(msg, dest=0):
    COM.send(msg, dest)


def recv(source=MPI.ANY_SOURCE):
    status = MPI.Status()
    msg = COM.recv(source=source, status=status)
    return msg, status


def sndrcv(msg, with_rank=0):
    COM.send(msg, with_rank)
    return COM.recv(source=with_rank)


def scatter(sendobj=None, recvobj=None, root=0):
    COM.scatter(sendobj, recvobj, root)


class MasterScheduler(Scheduler):

    def __init__(self, *args, **kwargs):
        super(MasterScheduler, self).__init__(*args, **kwargs)

class MPIScheduler(Scheduler):

    def add_task(self, *args, **kwargs):
        return sndrcv({'cmd': 'add_task', 'args': args, 'kwargs': kwargs})

    def add_worker(self, *args, **kwargs):
        return sndrcv({'cmd': 'add_worker', 'args': args, 'kwargs': kwargs})

    def get_work(self, *args, **kwargs):
        return sndrcv({'cmd': 'get_work', 'args': args, 'kwargs': kwargs})

    def ping(self, *args, **kwargs):
        return True
        #return sndrcv({'cmd': 'ping', 'args': args, 'kwargs': kwargs})


class MPIWorker(Worker):

    def __init__(self, scheduler=MPIScheduler(), worker_id=None,
                 worker_processes=1, assistant=False, **kwargs):

        self.worker_processes = int(worker_processes)
        self._worker_info = self._generate_worker_info()

        if not worker_id:
            worker_id = 'Worker(%s)' % ', '.join(['%s=%s' % (k, v) for k, v in self._worker_info])

	self._config = worker(**kwargs)

        self._id = worker_id
        self._scheduler = scheduler
        self._assistant = assistant

        self.host = socket.gethostname()
        self._scheduled_tasks = {}
        self._suspended_tasks = {}

	self._add_task_history = []
	self._batch_running_tasks = {}
	self._stop_requesting_work = False
	self._get_work_response_history = []
	self._batch_families_sent = set()

        self._first_task = None

        self.add_succeeded = True
        self.run_succeeded = True
        self.unfulfilled_counts = collections.defaultdict(int)

        self._task_result_queue = multiprocessing.Queue()
        self._running_tasks = {}

    def _generate_worker_info(self):
        args = super(MPIWorker, self)._generate_worker_info()
        args += [('rank', COM.Get_rank())]
        return args


class MasterMPIWorker(MPIWorker):

    def __init__(self, scheduler=MasterScheduler(), worker_id=None,
                 worker_processes=1, **kwargs):
        super(MasterMPIWorker, self).__init__(scheduler=scheduler, 
                                              worker_id=worker_id,
                                              worker_processes=worker_processes,
					      **kwargs)
        self._task_status = {}

    def _check_complete(self, task):
        return self._task_status.setdefault(task.task_id, task.complete())

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass

    def run(self):
        # Here we block to wait for all workers to be ready for
        # initialisation.  At this stage the MasterMPIWorker object
        # (i.e., self) has checked if tasks are already complete. This
        # allows the `SlaveMPIWorker`s to query the `MasterMPIWorker` to
        # see if the task is complete instead of thrashing the
        # (distributed) file system.

        log.debug('Synchronising with slaves')
        COM.Barrier()

        slaves_alive = COM.Get_size() - 1  # minus the master

        while slaves_alive > 0:
            msg, status = recv()
            cmd, args, kwargs = msg['cmd'], msg['args'], msg['kwargs']

            try:  # to pass the message to the master scheduler
                func = getattr(self._scheduler, cmd)
                result = func(*args, **kwargs)
                send(result, status.source)
            except AttributeError:
                if cmd == 'check_complete':
                    try:
                        task = self._scheduled_tasks[args[0]]
                        is_complete = self._check_complete(task)
                    except KeyError:
                        is_complete = True
                    send(is_complete, status.source)
                elif cmd == 'task_status':
                    send(self._task_status, status.source)
                elif cmd == 'stop':
                    slaves_alive -= 1

        return True


class SlaveMPIWorker(MPIWorker):

    def __init__(self, scheduler=MPIScheduler(), worker_id=None,
                 worker_processes=1, **kwargs):

        # Here we block to allow the MasterMPIWorker to check the
        # completion status of all tasks (see `_check_complete`). This
        # is to stop SlaveMPIWorkers from thrashing the (distributed)
        # file system.

        log.debug('Slave %i waiting to sync with Master', COM.Get_rank())
        COM.Barrier()

        self._task_status = {}
        send({'cmd': 'task_status', 'args': [], 'kwargs': None})
        result, status = recv(source=0)
        self._task_status.update(result)

        log.debug("Slave %i locally updated task status.", COM.Get_rank())

        # Now go ahead and initialise.

	super(SlaveMPIWorker, self).__init__(scheduler=scheduler,
                                             worker_id=worker_id,
                                             worker_processes=worker_processes,
					     **kwargs)

    def _check_complete(self, task):
        if self._task_status is None:
            return task.complete()
        else:
            return self._task_status[task.task_id]

    def run(self):
        log.debug("Slave %i is now allowed to check_complete", COM.Get_rank())
        self._task_status = None
        return super(SlaveMPIWorker, self).run()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        send({'cmd': 'stop', 'args': [self._id], 'kwargs': None}, 0)


class WorkerSchedulerFactory(object):

    def __init__(self, task_history=None):
        if COM.Get_rank() > 0:
            self.scheduler = MPIScheduler()
            self.worker = SlaveMPIWorker(self.scheduler)
        else:  # on master
            self.scheduler = MasterScheduler(task_history_impl=task_history)
            self.worker = MasterMPIWorker(self.scheduler)

    def create_local_scheduler(self):
        return self.scheduler

    def create_remote_scheduler(self, host, port):
        return NotImplemented

    def create_worker(self, scheduler, worker_processes=None, assistant=False):
        return self.worker


def run(tasks, task_history=None):
    sch = WorkerSchedulerFactory(task_history=task_history)
    build(tasks, worker_scheduler_factory=sch, local_scheduler=True)
