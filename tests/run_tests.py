import os
import sys

sys.path.append("../lcstack")
import argparse
import logging

from lcstack.cli import Client
from lcstack import set_config_root


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG_ROOT = "tests/configs/"


class CmdArgs:
    def __init__(
        self,
        config,
        init,
        name=None,
        query=None,
        input_key=None,
        output_key=None,
        kwargs={},
    ):
        self.name = name or init
        self.config = config
        self.init = init
        self.query = query
        self.input_key = input_key
        self.output_key = output_key
        self.kwargs = kwargs

    def as_args(self):
        args = [
            "-c",
            self.config,
            "-i",
            self.init,
        ]
        if self.query:
            args.extend(["-q", self.query])
        if self.input_key:
            args.extend(["-I", self.input_key])
        if self.output_key:
            args.extend(["-O", self.output_key])
        if self.kwargs:
            parts = [f"{k}={v}" for k, v in self.kwargs.items()]
            args.extend(["-a"] + parts)
        return args


def parse_tests_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help="specify config file, if not set, run with all test configs",
    )
    parser.add_argument(
        "-i", "--init", type=str, required=False, default=None, help="specify test name"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", help="list all test configs"
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        required=False,
        default=None,
        help="specify root config directory",
    )
    args = parser.parse_args()
    return args


def _append_results(collector, config, name, response=None, error=None, error_msg=None):
    collector.append(
        {
            "config": config,
            "name": name,
            "response": response,
            "error": error,
            "error_msg": error_msg,
        }
    )


# an example
tests = [
    CmdArgs(
        "graph_simple_as_agent.yaml",
        "graph_simple",
        query="tell me 2 joke about dog, but one at a time.",
        input_key="question",
    ),
]

test_files = [
    "basics.yaml",
    "chains_01.yaml",
    "workflows.yaml",
]


def _log(
    logger: logging.Logger,
    level,
    msg,
    args,
    exc_info=None,
    extra=None,
    stack_info=False,
    stacklevel=1,
) -> logging.LogRecord:
    """
    Low-level logging routine which creates a LogRecord and then calls
    all the handlers of this logger to handle the record.
    """
    sinfo = None
    try:
        fn, lno, func, sinfo = logger.findCaller(stack_info, stacklevel)
    except ValueError:  # pragma: no cover
        fn, lno, func = "(unknown file)", 0, "(unknown function)"
    else:  # pragma: no cover
        fn, lno, func = "(unknown file)", 0, "(unknown function)"
    if exc_info:
        if isinstance(exc_info, BaseException):
            exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
        elif not isinstance(exc_info, tuple):
            exc_info = sys.exc_info()
    record = logger.makeRecord(
        logger.name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
    )
    return record


def _format(record):
    return logging.lastResort.format(record)


def run_test(cmd_args: CmdArgs, collector: list[dict]):
    args = cmd_args.as_args()
    print("cmd_args: ", args)
    client = Client(args)
    inputs = client.query
    try:
        response = client.complete(inputs=inputs)
        _append_results(collector, cmd_args.config, cmd_args.name, response=response)
        logger.info(f"Assistant:\n{response}")
    except Exception as e:
        rec = _log(
            logger, logging.ERROR, e, args=[], exc_info=e, stack_info=True, stacklevel=3
        )
        _append_results(
            collector, cmd_args.config, cmd_args.name, error=e, error_msg=_format(rec)
        )
        logger.handle(rec)


def run_tests(cmds: list[CmdArgs], init: str, collector: list[dict]):
    if cmds:
        if init and init not in [t.name for t in cmds]:
            raise ValueError(f"Invalid init: {init}")
        for t in [t for t in cmds if init is None or init == t.name]:
            print(f"========= {t.config}::{t.name} =========")
            run_test(t, collector)
    else:
        for t in tests:
            print(f"========= {t.config}::{t.name} =========")
            run_test(t, collector)

    # print summary with colors
    print()
    print("========== Summary ==========")
    # count the number of successes and failures
    successes = len([r for r in collector if not r["error"]])
    failures = len([r for r in collector if r["error"]])
    print(f" Successes: {successes}")
    print(f"Failures: {failures}")
    print("========== Details ==========")
    for r in collector:
        if r["error"]:
            print(f"--> {r['config']}::{r['name']}: {r['error']}")


def load_test_file(file_path):
    import yaml

    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    return [CmdArgs(**t) for t in config]


def run_tests_with_files(file_path=None, init=None):
    if file_path:
        files = [file_path]
    else:
        test_dir = os.path.dirname(__file__)
        files = [os.path.join(test_dir, t) for t in test_files]

    collector = []
    for f in files:
        cmds = load_test_file(f)
        run_tests(cmds, init, collector)


def list_tests_with_files(
    file_path=None,
):
    if file_path:
        files = [file_path]
    else:
        test_dir = os.path.dirname(__file__)
        files = [os.path.join(test_dir, t) for t in test_files]

    for f in files:
        cmds = load_test_file(f)
        for t in cmds:
            print(f"{t.config}::{t.name}")


if __name__ == "__main__":
    app_args = parse_tests_args()
    set_config_root(app_args.root or DEFAULT_CONFIG_ROOT)
    init = app_args.init

    if app_args.list:
        list_tests_with_files(app_args.config)
        sys.exit(0)

    run_tests_with_files(app_args.config, init)
