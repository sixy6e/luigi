# -*- coding: utf-8 -*-
#
# Copyright 2012-2015 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""PBS batch system Tasks.

Borrowed heavily from the SunGrid Engine (SGE batch system Tasks code),
https://github.com/spotify/luigi/blob/master/luigi/contrib/sge.py.


Designed for use with a PBSPro job submission and scheduling program
available on HPC's such as those at the National Computation Infrastructure,
(NCI) http://nci.org.au/.
Jobs are submitted to the PBS queue using the ``qsub`` command and can be
monitored using ``qstat``.
The idea is to use luigi to submit, monitor, and chain various jobs together
in an easier manner than using BASH directly.  The various jobs can require
differing amounts of resources, and be self-contained luigi task workflows
themselves.

An environment for your job is typically setup using an ``init`` script,
containing various ``module load`` commands. As this can get complicated,
we'll leave that up to a user, and enforce the use of an ``init`` script
that will initialise your own environment, and execute your program.
The PBSTask can monitor the job, and kick of subsequent jobs, or other
jobs in parallel.

To run luigi workflows on an SGE cluster, subclass
:class:`luigi.contrib.sge.SGEJobTask` as you would any :class:`luigi.Task`,
but override the ``work()`` method, instead of ``run()``, to define the job
code. Then, run your Luigi workflow from the master node, assigning > 1
``workers`` in order to distribute the tasks in parallel across the cluster.

The following is an example usage (and can also be found in ``sge_tests.py``)

.. code-block:: python

    import logging
    import luigi
    import os
    from luigi.contrib.sge import SGEJobTask

    logger = logging.getLogger('luigi-interface')


    class TestJobTask(SGEJobTask):

        i = luigi.Parameter()

        def work(self):
            logger.info('Running test job...')
            with open(self.output().path, 'w') as f:
                f.write('this is a test')

        def output(self):
            return luigi.LocalTarget(os.path.join('/home', 'testfile_' + str(self.i)))


    if __name__ == '__main__':
        tasks = [TestJobTask(i=str(i), n_cpu=i+1) for i in range(3)]
        luigi.build(tasks, local_scheduler=True, workers=3)


The ``n-cpu`` parameter allows you to define different compute resource
requirements (or slots, in SGE terms) for each task. In this example, the
third Task asks for 3 CPU slots. If your cluster only contains nodes with
2 CPUs, this task will hang indefinitely in the queue. See the docs for
:class:`luigi.contrib.sge.SGEJobTask` for other SGE parameters. As for any
task, you can also set these in your luigi configuration file as shown below.
The default values below were matched to the values used by MIT StarCluster,
an open-source SGE cluster manager for use with Amazon EC2::

    [SGEJobTask]
    shared-tmp-dir = /home
    parallel-env = orte
    n-cpu = 2


"""


# This extension is modeled after the hadoop.py approach.
#
# Implementation notes
# The procedure:
# - Pickle the class
# - Construct a qsub argument that runs a generic runner function with the path to the pickled class
# - Runner function loads the class from pickle
# - Runner function hits the work button on it

import os
from os.path import join as pjoin, exists
import subprocess
import time
import sys
import logging

import luigi
#import luigi.hadoop
#from luigi.contrib import sge_runner

logger = logging.getLogger('luigi-interface')
logger.propagate = 0

POLL_TIME = 5  # decided to hard-code rather than configure here


def _parse_qstat_state(qstat_out):
    """Parse "state" column from `qstat` output for given job_id

    Returns state for the *first* job matching job_id. Returns 'u' if
    `qstat` output is empty or job_id is not found.

    """
    lines = qstat_out.split('\n')
    state = lines[2].strip().split(' ')[-2]
    return state


def _parse_qsub_job_id(qsub_out):
    """Parse job id from qsub output string.

    Assume format:

        "id.server_name"

    """
    return qsub_out.split('.')[0]


def _build_qsub_command(job_name, stdout=None, stderr=None, project=None,
                        walltime=None, memory=None, queue=None, ncpus=None,
                        job_script=None):
    """Submit shell command to PBS queue via `qsub`"""
    lspec = "walltime={walltime},mem={memory}GB,ncpus={ncpus}"
    cmd = ['qsub',
           '-o', stdout,
           '-e', stderr,
           '-P', project,
           '-q', queue,
           '-l', lspec.format(walltime=walltime, mem=memory, ncpus=ncpus),
           '-N', job_name,
           '{}'.format(job_script)]
    return cmd


class PBSJobTask(luigi.Task):

    """Base class for submitting a job to the PBS queue.

    Override ``work()`` (rather than ``run()``) with your job code.

    Parameters:

    - n_cpu: Number of CPUs (or "slots") to allocate for the Task. This
          value is passed as ``qsub -pe {pe} {n_cpu}``
    - job_name: Exact job name to pass to qsub.
    - poll_time: the length of time to wait in order to poll qstat

    TODO: investigate if we should include job_name_format as an option
    - job_name_format: String that can be passed in to customize the job name
        string passed to qsub; e.g. "Task123_{task_family}_{n_cpu}...".
    """

    n_cpu = luigi.IntParameter(default=16, significant=False)
    memory = luigi.IntParameter(default=32, significant=False,
                                description="Memory in Gb to be requested.")
    project = luigi.Parameter(significant=False)
    walltime = luigi.Parameter(significant=False,
                               description="Requested job time in HH:MM:SS.")
    queue = luigi.Parameter(default='normal', significant=False,
                            description='The queue to submit the job into')
    job_script = luigi.Parameter(significant=False,
                                 description=("Full file path name to the "
                                              "script that qsub will execute."))
    job_name = luigi.Parameter(significant=False, default=None,
                               description="Explicit job name given via qsub.")
    poll_time = luigi.IntParameter(significant=False, default=POLL_TIME,
                                   description=("Specify the wait time to "
                                                "poll qstat for the job "
                                                "status."))
    log_directory = luigi.Parameter(significant=False)
    # job_name_format = luigi.Parameter(
    #     significant=False, default=None, description="A string that can be "
    #     "formatted with class variables to name the job with qsub.")

    stdout = 'blank'
    stderr = 'blank'
    _completed = False


    def _fetch_task_failures(self):
        # TODO: catch if we have no stdout file.
        if not exists(self.stdout):
            logger.info('No stdout file')
            return True
        with open(self.stdout, "r") as src:
            errors = src.readlines()
        exit_status = int(errors[-8].strip().split(' ')[-1])

        failed = True if exit_status != 0 else False
        return failed

    def run(self):
        self._run_job()

    def _run_job(self):
        # Check that we have a log directory to write to
        if not exists(self.log_directory):
            os.makedirs(self.log_directory)

        # Build qsub submit command
        self.stdout = pjoin(self.log_directory, self.job_name + '.stdout')
        self.stderr = pjoin(self.log_directory, self.job_name + '.stderr')
        submit_cmd = _build_qsub_command(self.job_name, self.stdout,
                                         self.stderr, self.project,
                                         self.walltime, self.memory,
                                         self.queue, self.n_cpu,
                                         self.job_script)
        logger.debug('qsub command: \n' + submit_cmd)

        # Submit the job and grab job ID
        qsub_output = subprocess.check_output(submit_cmd)
        self.job_id = _parse_qsub_job_id(qsub_output)
        logger.debug("Submitted job to qsub with response:\n" + qsub_output)

        self._track_job()


    def _track_job(self):
        while True:
            # Sleep for a little bit
            time.sleep(self.poll_time)

            # check the job status
            try:
                qstat_out = subprocess.check_output(['qstat',
                                                     '-x',
                                                     self.job_id])
                pbs_status = _parse_qstat_state(qstat_out)

            except subprocess.CalledProcessError as e:
                # potentially an error connecting to the pbs server
                logger.info(e.output)
                continue

            if pbs_status == 'R':
                logger.info('Job is running...')
            elif pbs_status == 'Q':
                logger.info('Job is pending...')
            elif pbs_status == 'H':
                logger.info('Job is held...')
            elif pbs_status == 'S':
                logger.info('Job is suspended...')
            elif pbs_status == 'E':
                logger.info('Job is exiting...')
            elif 'F' in pbs_status:
                logger.info('Job has finished')
                # check the stdout for exit status
                failed = self._fetch_task_failures()
                if failed:
                    # raise an exception or something to halt the workflow
                    logger.error('Job has FAILED')
                    self.failed = True
                else:
                    logger.info('Job is done')
                    self.failed = False
                    self._completed = True
                break
            else:
                logger.info('Job status is UNKNOWN!')
                logger.info('Status is : %s' % pbs_status)
                raise Exception("job status isn't one of ['R', 'Q', 'H', 'S', 'E', 'F']: %s" % pbs_status)


    def complete(self):
        return self._completed



# class LocalSGEJobTask(SGEJobTask):
#     """A local version of SGEJobTask, for easier debugging.
#
#     This version skips the ``qsub`` steps and simply runs ``work()``
#     on the local node, so you don't need to be on an SGE cluster to
#     use your Task in a test workflow.
#     """
#
#     def run(self):
#         self.work()
