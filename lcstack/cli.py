import argparse
import os
import uuid


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def _try_parse_val(val):
    if val is None:
        return None
    try:
        from ast import literal_eval

        parsed_val = literal_eval(val)
        # NOTE: only support from str to dict, list.
        # if int, float, bool?
        if isinstance(parsed_val, (dict, list)):
            return parsed_val
    except ValueError:
        pass
    return val


class Client:
    def __init__(self, args=None) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="config file under config root directory",
        )
        self.parser.add_argument(
            "-i", "--init", type=str, required=True, help="configured initializer name"
        )
        self.parser.add_argument(
            "-q",
            "--query",
            type=str,
            default=None,
            help="query string, if input is a single string, or a dict with one key",
        )
        self.parser.add_argument(
            "-a",
            "--kwargs",
            nargs="*",
            action=ParseKwargs,
            default=None,
            help="input's key-value pairs if it's dict",
        )
        self.parser.add_argument(
            "-t", "--template", type=str, default=None, help="specify template file"
        )
        self.parser.add_argument(
            "-s",
            "--settings",
            type=str,
            default=None,
            help="settings file to load, default to config root settings.yaml",
        )
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="verbose mode which prints more info",
        )
        self.parser.add_argument(
            "-T",
            "--trace",
            action="store_true",
            help="trace mode which outputs to a json file",
        )
        self.parser.add_argument(
            "-I",
            "--input_key",
            type=str,
            default=None,
            help="input key if the input is a dict with one key",
        )
        self.parser.add_argument(
            "-O",
            "--output_key",
            type=str,
            default=None,
            help="the key of output which will be displayed if the output is a dict",
        )
        self.parser.add_argument(
            "-G",
            "--graph",
            type=str,
            default=None,
            help="graph file to save the graph of this component",
        )
        self.parser.add_argument(
            "-S", "--session", type=str, default=None, help="specify session id"
        )
        self.parser.add_argument(
            "--langsmith",
            action="store_true",
            help="use langsmith for tracing, must specify `LANGCHAIN_API_KEY` environment variable",
        )
        self.parser.add_argument(
            "--original",
            action="store_true",
            help="use original component, without wrapper of input or output preprocessing",
        )
        self.parser.add_argument(
            "--stream",
            action="store_true",
            help="stream output mode, just for experiment",
        )

        self.args = self.parser.parse_args(args)

        self.config = self.args.config
        self.init = self.args.init
        self.template = self.args.template
        self.settings = self.args.settings
        self.query = self.args.query
        self.kwargs = self.args.kwargs or {}
        self.verbose = self.args.verbose
        self.trace = self.args.trace
        self.input_key = self.args.input_key
        self.output_key = self.args.output_key
        self.graph = self.args.graph
        self.session = self.args.session
        self.langsmith = self.args.langsmith
        # use original component, without wrapper of input or output preprocessing
        self.original = self.args.original
        self.stream = self.args.stream

    def _save_image(self, image):
        if image is None:
            return
        # save into a file
        file_path = (
            self.graph
            if self.graph.endswith(".png")
            else os.path.join(self.graph, f"{self.init}.png")
        )
        with open(file_path, "wb") as f:
            f.write(image)

    def _default_exit_func(self, outputs):
        from langchain_core.messages import BaseMessage

        if isinstance(outputs, dict) and self.output_key in outputs:
            return outputs.get(self.output_key)
        elif isinstance(outputs, BaseMessage) and self.output_key in ["content"]:
            return outputs.content
        return outputs

    def complete(self, inputs=None, input_key=None):
        input_key = input_key or self.input_key
        inputs = _try_parse_val(inputs) or self.query
        inputs_dict = self.kwargs or {}
        for k, v in inputs_dict.items():
            inputs_dict[k] = _try_parse_val(v)
        if input_key is not None:
            inputs_dict[input_key] = inputs
        inputs = inputs_dict or inputs

        if self.verbose:
            print("inputs: ", inputs)

        from .components.tracers.console import ConsoleCallbackHandler
        from .base import LcStackBuilder
        from .tracers import JsonCallbackHandler

        lcs = (
            LcStackBuilder.from_yaml(self.config)
            .with_env()
            .with_secret()
            .with_settings(self.settings or True)
            # .pprint(self.verbose)
            # .with_template(self.template)
            .build()
        )

        # should be clear their inputs for the original components
        if self.original:
            initializer = lcs._get_root_initializer(self.init)
            # Runnable or lc original component
            # NOTE: no invoke method if it's not a Runnable instance, like document loader, etc.
            app = initializer.build(name="app").build_original()
        else:
            # Container, the invoke
            app = lcs.build(self.init)

        if self.graph is not None:
            self._save_image(lcs.draw_graph_png())

        # TODO: put here, just to make sure the .env is loaded
        if self.langsmith:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if "LANGCHAIN_API_KEY" not in os.environ:
                raise ValueError(
                    "To use Langsmith tracing, please set your LANGCHAIN_API_KEY environment variable"
                )
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"

        callbacks = []
        if self.verbose:
            callbacks.append(
                ConsoleCallbackHandler(
                    callbacks=[
                        "llm",
                        "chain",
                        "agent",
                        "retriever",
                    ]
                )
            )  # , "chain", "agent", "retriever", "custom_event", "retry"
        if self.trace:
            callbacks.append(JsonCallbackHandler())
        session_id = self.session or f"{self.init}_{uuid.uuid4().hex[:8]}"
        run_config = {
            "thread_id": session_id,
            "callbacks": callbacks,
        }  # {"max_iterations": 100, "session_id": session_id, "thread_id": session_id}
        if self.stream:
            runnable = app.build_original()
            response = {}
            run_config["configurable"] = {"stream_mode": "values"}
            for chunk in runnable.stream(inputs, run_config):
                print("-->", chunk)
                response.update(list(chunk.values())[0])
            response = self._default_exit_func(response)
        else:
            response = app.invoke(inputs, run_config)
            response = self._default_exit_func(response)
        return response
