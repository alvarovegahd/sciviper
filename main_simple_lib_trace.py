# General imports and variables, as well as config
import ast
import math
import sys
import time

import requests
import torch.multiprocessing as mp
from joblib import Memory
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.pretty import pprint
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich import print
from rich.markup import escape as rich_escape

from IPython.display import update_display, clear_output, display
from PIL import Image
import matplotlib.pyplot as plt

from configs import config
from utils import show_single_image

from IPython.display import update_display, clear_output
from IPython.core.display import HTML

from vision_processes import forward, finish_all_consumers  # This import loads all the models. May take a while
from image_patch import *
from video_segment import *
from datasets_local.my_dataset import MyDataset


##### For Trace
from src.Trace.opto.trace import node, Module, GRAPH
from src.Trace.opto.trace import model, bundle, ExecutionError
# import inspect
###############

def get_thing_to_show_codetype(codeline):
    # can output either a list of things to show, or a single thing to show
    things_to_show = []
    if codeline.startswith("if"):
        condition, rest = codeline[3:].split(":", 1)
        codeline = f"if {condition}:{rest}"
        code_type = "if"

        operators = ['==', '!=', '>=', '<=', '>', '<']
        things_to_show = []
        for op in operators:
            if op in condition:
                things_to_show = [x.strip() for x in condition.split(op)]
                # print(things_to_show)
                break
        # things_to_show.append(thing_to_show)
        thing_to_show = things_to_show + [condition.strip()]

    elif codeline.startswith("for"):
        code_type = 'for'
        thing_to_show = codeline.split("for ")[1].split(" in ")[0]

    elif codeline.startswith("return"):
        thing_to_show = codeline.split("return ")[1]
        code_type = 'return'

    elif ' = ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' = ')[0]
    elif ' += ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' += ')[0]
    elif ' -= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' -= ')[0]
    elif ' *= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' *= ')[0]
    elif ' /= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' /= ')[0]

    elif '.append(' in codeline:
        code_type = 'append'
        thing_to_show = codeline.split('.append(')[0] + '[-1]'
    elif '.add(' in codeline:
        code_type = 'add'
        thing_to_show = codeline.split('.add(')[0]

    elif '.sort(' in codeline:
        code_type = 'sort'
        thing_to_show = codeline.split('.sort(')[0]

    elif codeline.startswith("elif") or codeline.startswith("else"):
        thing_to_show = None
        code_type = 'elif_else'
    else:
        thing_to_show = None
        code_type = 'other'

    if isinstance(thing_to_show, list):
        thing_to_show = [thing if not (thing.strip().startswith("'") and thing.strip().endswith("'"))
                         else thing.replace("'", '"') for thing in thing_to_show if thing is not None]
    elif isinstance(thing_to_show, str):
        thing_to_show = thing_to_show if not (thing_to_show.strip().startswith("'") and
                                              thing_to_show.strip().endswith("'")) else thing_to_show.replace("'", '"')
    return thing_to_show, code_type


def split_codeline_and_indent_level(codeline):
    # print("======")
    # print(codeline)
    # print("======")
    origlen = len(codeline)
    codeline = codeline.lstrip()
    indent = origlen - len(codeline)
    indent = " " * indent
    return codeline, indent


def show_one_image(image, ax):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.dtype == torch.float32:
            image = image.clamp(0, 1)
        image = image.squeeze(0).permute(1, 2, 0)
    ax.imshow(image)


def CodexAtLine(lineno, syntax, time_wait_between_lines=1.):
    syntax._stylized_ranges = []
    syntax.stylize_range('on red', (lineno + 1, 0), (lineno + 1, 80))
    time.sleep(time_wait_between_lines)

def load_image(path):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        image = transforms.ToTensor()(image)
    else:
        image = Image.open(path)
        image = transforms.ToTensor()(image)
    return image

# this works with qwen2.5-coder
def parse_function(code_block):
    # Split by triple backticks and get the content
    parts = code_block.split('```')
    if len(parts) >= 3:
        function_code = parts[1]
    else:
        function_code = code_block

    # Remove the "python" line if present
    lines = function_code.split('\n')
    if lines[0].strip().lower() == 'python':
        lines = lines[1:]

    # Join the lines and strip whitespace
    parsed_function = '\n'.join(lines).strip()

    return parsed_function

def is_not_fig(x):
    if x is None:
        return True
    elif isinstance(x, (str, float, int)):
        return True
    elif isinstance(x, (list, tuple)):
        return all([is_not_fig(xx) for xx in x])
    elif isinstance(x, dict):
        return all([is_not_fig(xx) for xx in x.values()])
    return False

def show_single_image(im):
    im = Image.fromarray((im.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))
    im.copy()
    im.thumbnail((400, 400))
    display(im)


def show_all(lineno, value, valuename, fig=None, usefig=True, disp=True,
                console_in=None, time_wait_between_lines=None, lastlineno=[-1]):
        time.sleep(0.1)  # to avoid race condition!

        assert console_in is not None

        thing_to_show = value

        if lineno is not None and lineno != lastlineno[0]:
            console_in.rule(f"[bold]Line {lineno}[/bold]", style="chartreuse2")
            lastlineno[0] = lineno  # ugly hack

        if usefig:
            plt.clf()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
        if isinstance(thing_to_show, Image.Image):
            if valuename:
                console_in.print(f'{rich_escape(valuename)} = ')
            show_one_image(thing_to_show, ax)
        elif str(type(thing_to_show)) == "<class 'image_patch.ImagePatch'>":
            if valuename:
                console_in.print(f'{rich_escape(valuename)} = ')
            show_one_image(thing_to_show.cropped_image, ax)
        elif isinstance(thing_to_show, list) or isinstance(thing_to_show, tuple):
            if len(thing_to_show) > 0:
                for i, thing in enumerate(thing_to_show):
                    disp_ = disp or i < len(thing_to_show) - 1
                    show_all(None, thing, f"{rich_escape(valuename)}[{i}]", fig=fig, disp=disp_, usefig=usefig, console_in=console_in)
                return
            else:
                console_in.print(f"{rich_escape(valuename)} is empty")
        elif isinstance(thing_to_show, dict):
            if len(thing_to_show) > 0:
                for i, (thing_k, thing_v) in enumerate(thing_to_show.items()):
                    disp_ = disp or i < len(thing_to_show) - 1
                    show_all(None, thing_v, f"{rich_escape(valuename)}['{thing_k}']", fig=fig, disp=disp_, usefig=usefig, console_in=console_in)
                return
            else:
                console_in.print(f"{rich_escape(valuename)} is empty")
        else:
            console_in.print(f"{rich_escape(valuename)} = {thing_to_show}")
            if time_wait_between_lines is not None:
                time.sleep(time_wait_between_lines / 2)
            return

        # display small
        if usefig:
            fig.set_size_inches(2, 2)
            if disp:
                display(fig)

def inject_saver(code, show_intermediate_steps, syntax=None, time_wait_between_lines=None, console=None):
    assert console is not None

    injected_function_name = 'show_all'
    if injected_function_name in code:
        return code
    code = code.split("\n")
    newcode = []
    for n, codeline in enumerate(code):
        codeline, indent = split_codeline_and_indent_level(codeline)

        if codeline.startswith('#') or codeline == '':  # this will cause issues if you have lots of comment lines
            continue
        if '#' in codeline:
            codeline = codeline.split('#')[0]

        thing_to_show, code_type = get_thing_to_show_codetype(codeline)

        if code_type in ('assign', 'append', 'if', 'return', 'for', 'sort', 'add'):
            if '\'' in codeline:
                codeline.replace('\'', '\\\'')

            if show_intermediate_steps:
                escape_thing = lambda x: x.replace("'", "\\'")
                injection_string_format = \
                    lambda \
                        thing: f"{indent}{injected_function_name}(lineno={n},value=({thing}),valuename='{escape_thing(thing)}'," \
                            f"fig=my_fig,console_in=console,time_wait_between_lines=time_wait_between_lines); " \
                            f"CodexAtLine({n},syntax=syntax,time_wait_between_lines=time_wait_between_lines)"
            else:
                injection_string_format = lambda thing: f"{indent}CodexAtLine({n},syntax=syntax," \
                                                        f"time_wait_between_lines=time_wait_between_lines)"

            extension_list = []
            if isinstance(thing_to_show, list):
                injection_string_list = [injection_string_format(f"{thing}") for thing in thing_to_show]
                extension_list.extend(injection_string_list)
            elif code_type == 'for':
                injection_string = injection_string_format(f"{thing_to_show}")
                injection_string = " " * 4 + injection_string
                extension_list.append(injection_string)
            else:
                extension_list.append(injection_string_format(f"{thing_to_show}"))

            if code_type in ('if', 'return'):
                extension_list = extension_list + [f"{indent}{codeline}"]
            else:
                extension_list = [f"{indent}{codeline}"] + extension_list

            newcode.extend(extension_list)

        elif code_type == 'elif_else':
            newcode.append(f"{indent}{codeline}")
        else:
            newcode.append(f"{indent}{codeline}")
    return "\n".join(newcode)

class ViperGPT(Module):
    def __init__(self):
        # Initialize class-level variables
        self.time_wait_between_lines = 0.5
        self.console = Console(highlight=False, force_terminal=False)
        self.cache = Memory('cache/' if config.use_cache else None, verbose=0)
        mp.set_start_method('spawn', force=True)

    def get_code_trace(self, query):
        print("INSIDE GET CODE WITH TRACE!!!")
        print(f'{query=}')
        code = self.generate_execute_command(query)
        code = f'def execute_command(image, my_fig, time_wait_between_lines, syntax):' + code

        code_string = code.detach()._data

        code_for_syntax = code_string.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
        syntax_1 = Syntax(code_for_syntax, "python", theme="monokai", line_numbers=True, start_line=0)
        self.console.print(syntax_1)
        code_string = ast.unparse(ast.parse(code_string))
        code_for_syntax_2 = code_string.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
        syntax_2 = Syntax(code_for_syntax_2, "python", theme="monokai", line_numbers=True, start_line=0)
        return code, syntax_2

    @staticmethod
    @bundle(trainable=True, catch_execution_error=True, allow_external_dependencies=True)
    def generate_execute_command(query):
        """Generate execute_command function body based on the query"""
        code_body = "\n\treturn None"
        return code_body

    def execute_code(self, code, im, show_intermediate_steps=True):
        code, syntax = code
        print(f'{code=}')
        code_line = inject_saver(code.data, show_intermediate_steps, syntax,
                                    self.time_wait_between_lines, self.console)

        display(HTML("<style>.output_wrapper, .output {height:auto !important; max-height:1000000px;}</style>"))

        with Live(Padding(syntax, 1), refresh_per_second=10, console=self.console, auto_refresh=True) as live:
            my_fig = plt.figure(figsize=(4, 4))
            try:
                exec(compile(code_line, 'Codex', 'exec'), globals())
                result = execute_command(im, my_fig, self.time_wait_between_lines, syntax)
            except Exception as e:
                print(f"Encountered error {e} when trying to run with visualizations. Trying from scratch.")
                try:
                    exec(compile(code, 'Codex', 'exec'), globals())
                    result = execute_command(im, my_fig, self.time_wait_between_lines, syntax)
                except Exception as e:
                    print(f'Encountered the following exception: {e}')
                    result = f'Exception: {e}'
                    return result

            plt.close(my_fig)

        f = None
        usefig = False
        if not is_not_fig(result):
            f = plt.figure(figsize=(4, 4))
            usefig = True

        self.console.rule(f"[bold]Final Result[/bold]", style="chartreuse2")
        show_all(None, result, 'Result', fig=f, usefig=usefig, disp=False,
                     console_in=self.console, time_wait_between_lines=0)

        return result


class ViperGPT_v2(Module):
    def __init__(self):
        # Initialize class-level variables
        self.time_wait_between_lines = 0.5
        self.console = Console(highlight=False, force_terminal=False)
        self.cache = Memory('cache/' if config.use_cache else None, verbose=0)
        mp.set_start_method('spawn', force=True)

    # def get_code_trace(self, query):
    #     print("INSIDE GET CODE WITH TRACE!!!")
    #     print(f'{query=}')
    #     code = self.generate_execute_command(query)
    #     code = f'def execute_command(image, my_fig, time_wait_between_lines, syntax):' + code

    #     code_string = code.detach()._data

    #     code_for_syntax = code_string.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
    #     syntax_1 = Syntax(code_for_syntax, "python", theme="monokai", line_numbers=True, start_line=0)
    #     self.console.print(syntax_1)
    #     code_string = ast.unparse(ast.parse(code_string))
    #     code_for_syntax_2 = code_string.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
    #     syntax_2 = Syntax(code_for_syntax_2, "python", theme="monokai", line_numbers=True, start_line=0)
    #     return code, syntax_2

    @bundle(trainable=True, catch_execution_error=True, allow_external_dependencies=True)
    def execute_command(self, image, query):
        """Execute the command based on the image and query"""
        image_patch = ImagePatch(image)
        muffin_patches = image_patch.find("muffin")
        kid_patches = image_patch.find("kid")

        if len(muffin_patches) == 0 or len(kid_patches) == 0:
            return "No muffins or kids found in the image."

        muffin_count = len(muffin_patches)
        kid_count = len(kid_patches)

        if muffin_count < kid_count:
            return "Not enough muffins for each kid to have one."

        muffins_per_kid = muffin_count // kid_count
        remaining_muffins = muffin_count % kid_count

        answer = f"Each kid can have {muffins_per_kid} muffin(s). "
        if remaining_muffins > 0:
            answer += f"There will be {remaining_muffins} muffin(s) left over."

        return answer

        # return "I don't know"

    def execute_code(self, query, im, show_intermediate_steps=True):
        # code, syntax = code
        # print(f'{code=}')
        # code_line = inject_saver(code.data, show_intermediate_steps, syntax,
        #                             self.time_wait_between_lines, self.console)

        display(HTML("<style>.output_wrapper, .output {height:auto !important; max-height:1000000px;}</style>"))

        # my_fig = plt.figure(figsize=(4, 4))
        try:
            result = self.execute_command(im, query)
            feedback = "It is wrong"
        except ExecutionError as e:
            print(f"Encountered error {e} when trying to run with visualizations. Trying from scratch.")
            result = e.exception_node
            feedback = result.data

        # plt.close(my_fig)

        f = None
        usefig = False
        if not is_not_fig(result):
            f = plt.figure(figsize=(4, 4))
            usefig = True

        self.console.rule(f"[bold]Final Result[/bold]", style="chartreuse2")
        show_all(None, result, 'Result', fig=f, usefig=usefig, disp=False,
                     console_in=self.console, time_wait_between_lines=0)

        return result