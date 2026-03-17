#!/usr/bin/env python3

import click
import csv
import json
from importlib.resources import files

import numpy as np
from jinja2 import Template

from eyeballer.model import EyeballModel, DATA_LABELS
from eyeballer.visualization import HeatMap


class _FloatEncoder(json.JSONEncoder):
    """Serialize numpy floats to plain Python floats."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def _parse_thresholds(ctx, param, value):
    """Click callback: parse 'label=value,...' into a {label: float} dict."""
    if not value:
        return {}
    result = {}
    for pair in value.split(','):
        pair = pair.strip()
        if '=' not in pair:
            raise click.BadParameter(f"Expected 'label=value', got '{pair}'")
        label, val = pair.split('=', 1)
        label = label.strip()
        if label not in DATA_LABELS:
            raise click.BadParameter(
                f"Unknown label '{label}'. Valid labels: {', '.join(DATA_LABELS)}")
        try:
            result[label] = float(val.strip())
        except ValueError:
            raise click.BadParameter(
                f"Value for '{label}' must be a float, got '{val.strip()}'")
    return result


@click.group(invoke_without_command=True)
@click.option('--weights', default="eyeballer-v3.weights.h5", type=click.Path(), help="Weights file for input/output")
@click.option('--summary/--no-summary', default=False, help="Print model summary at start")
@click.option('--seed', default=None, type=int, help="RNG seed for data shuffling and transformations, defaults to random value")
@click.pass_context
def cli(ctx, weights, summary, seed):
    model_kwargs = {"weights_file": weights,
                    "print_summary": summary,
                    "seed": seed}

    #  pass the model to subcommands
    ctx.ensure_object(dict)
    # We only pass the kwargs so we can be lazy and make the model later after the subcommand cli is parsed. This
    # way, the user doesn't have to wait for tensorflow if they are just calling --help on a subcommand.
    ctx.obj['model_kwargs'] = model_kwargs


@cli.command()
@click.option('--graphs/--no-graphs', default=False, help="Save accuracy and loss graphs to file")
@click.option('--epochs', default=20, type=int, help="Number of epochs")  # TODO better help string
@click.option('--batchsize', default=32, type=int, help="Batch size")  # TODO better help string
@click.pass_context
def train(ctx, graphs, batchsize, epochs):
    model = EyeballModel(**ctx.obj['model_kwargs'])
    model.train(print_graphs=graphs, batch_size=batchsize, epochs=epochs)


@cli.command()
@click.argument('screenshot')
@click.option('--heatmap', default=False, is_flag=True, help="Create a heatmap graph for the prediction")
@click.option('--threshold', default=.5, type=float,
              help="Confidence threshold applied to all labels (default: 0.5)")
@click.option('--thresholds', default=None, callback=_parse_thresholds, is_eager=False,
              help="Per-label thresholds, overrides --threshold for specified labels. "
                   "Format: label=value,... e.g. login=0.3,parked=0.7")
@click.option('--format', 'output_format', default='csv',
              type=click.Choice(['csv', 'json']), help="Output format for multi-file results (default: csv)")
@click.pass_context
def predict(ctx, screenshot, heatmap, threshold, thresholds, output_format):
    model = EyeballModel(**ctx.obj['model_kwargs'])
    results = model.predict(screenshot)

    if heatmap:
        # Generate a heatmap
        HeatMap(screenshot, model, threshold).generate()

    if not results:
        print("Error: Input file does not exist")
        return

    if len(results) == 1:
        if output_format == 'json':
            print(json.dumps(results[0], indent=2, cls=_FloatEncoder))
        else:
            print(results[0])
    else:
        if output_format == 'json':
            with open("results.json", "w") as f:
                json.dump(results, f, indent=2, cls=_FloatEncoder)
            print("Output written to results.json")
        else:
            with open("results.csv", "w", newline="") as csvfile:
                fieldnames = ["filename", "custom404", "login", "webapp", "oldlooking", "parked"]
                labelwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                labelwriter.writeheader()
                labelwriter.writerows(results)
            print("Output written to results.csv")

        buildHTML(processResults(results, threshold, thresholds))
        print("HTML written to results.html")


def processResults(results, threshold, thresholds=None):
    '''Filter the initial results dictionary and reformat it for use in JS.

        Keyword arguments:
        results -- dictionary output from predict function
        threshold -- default confidence threshold for all labels
        thresholds -- optional dict of per-label threshold overrides
    '''
    effective = {label: threshold for label in DATA_LABELS}
    if thresholds:
        effective.update(thresholds)

    jsResults = {}
    for result in results:
        positiveTags = []
        for label, label_info in result.items():
            if label == 'filename':
                pass
            elif label_info > effective.get(label, threshold):
                positiveTags.append(label)
        jsResults[result['filename']] = positiveTags
    return jsResults


def buildHTML(jsResults):
    '''Build HTML around the JS Dictionary that is passed from processResults.

        Keyword arguments:
        jsResults -- dictionary output from processResults function
    '''
    template_text = files("eyeballer").joinpath("prediction_output_template.html").read_text(encoding="utf-8")
    html_output = Template(template_text).render(jsResults=jsResults)

    with open('results.html', 'w') as f:
        f.write(html_output)


def pretty_print_evaluation(results):
    """Print a human-readable summary of the evaluation"""
    # We use 4.2% to handle all the way from "  0.00%" (7chars) to "100.00%" (7chars)
    for label in DATA_LABELS:
        print("{} Precision Score: {:4.2%}".format(label, results[label]['precision']))
        print("{} Recall Score: {:4.2%}".format(label, results[label]['recall']))
    print("'None of the above' Precision: {:4.2%}".format(results['none_of_the_above_precision']))
    print("'None of the above' Recall: {:4.2%}".format(results['none_of_the_above_recall']))
    print("All or nothing Accuracy: {:4.2%}".format(results['all_or_nothing_accuracy']))
    print("Overall Binary Accuracy: {:4.2%}".format(results['total_binary_accuracy']))
    print("Top 10 worst predictions: {}".format(results['top_10_worst'][1]))


@cli.command()
@click.option('--threshold', default=.5, type=float, help="Threshold confidence for labeling")
@click.pass_context
def evaluate(ctx, threshold):
    model = EyeballModel(**ctx.obj['model_kwargs'])
    results = model.evaluate(threshold)
    pretty_print_evaluation(results)


if __name__ == '__main__':
    cli()
