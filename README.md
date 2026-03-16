# Eyeballer

![Logo](/docs/eyeballer_logo.png)


Give those screenshots of yours a quick eyeballing.

Eyeballer is meant for large-scope network penetration tests where you need to find "interesting" targets from a huge set of web-based hosts. Go ahead and use your favorite screenshotting tool like normal (EyeWitness or GoWitness) and then run them through Eyeballer to tell you what's likely to contain vulnerabilities, and what isn't.

Give it a try live at: https://eyeballer.bishopfox.com

### Example Labels

| Old-Looking Sites | Login Pages |
| ------ |:-----:|
| ![Sample Old-looking Page](/docs/old-looking.png) | ![Sample Login Page](/docs/login.png) |

| Webapp | Custom 404's |
| ------ |:-----:|
| ![Sample Webapp](/docs/homepage.png) | ![Sample Custom 404](/docs/404.png) |

| Parked Domains |
| ------ |
| ![Sample Webapp](/docs/parked.png) |

## What the Labels Mean

**Old-Looking Sites**
Blocky frames, broken CSS, that certain "je ne sais quoi" of a website that looks like it was designed in the early 2000's. You know it when you see it. Old websites aren't just ugly, they're also typically super vulnerable. When you're looking to hack into something, these websites are a gold mine.

**Login Pages**
Login pages are valuable to pen testing, they indicate that there's additional functionality you don't currently have access to. It also means there's a simple follow-up process of credential enumeration attacks. You might think that you can set a simple heuristic to find login pages, but in practice it's really hard. Modern sites don't just use a simple input tag we can grep for.

**Webapp**
This tells you that there is a larger group of pages and functionality available here that can serve as surface area to attack. This is in contrast to a simple login page, with no other functionality. Or a default IIS landing page which has no other functionality. This label should indicate to you that there is a web application here to attack.

**Custom 404**
Modern sites love to have cutesy custom 404 pages with pictures of broken robots or sad looking dogs. Unfortunately, they also love to return HTTP 200 response codes while they do it. More often, the "404" page doesn't even contain the text "404" in it. These pages are typically uninteresting, despite having a lot going on visually, and Eyeballer can help you sift them out.

**Parked Domains**
Parked domains are websites that look real, but aren't valid attack surface. They're stand-in pages, usually devoid of any real functionality, consist almost entirely of ads, and are usually not run by our actual target. It's what you get when the domain specified is wrong or lapsed. Finding these pages and removing them from scope is really valuable over time.

## Setup

This fork requires **Python 3.11–3.13** and **TensorFlow ≥ 2.16** (Keras 3).

```
pip install -r requirements.txt
```

**Pretrained Weights**

Download `eyeballer-v3.weights.h5` from the [releases page](https://github.com/maxvaer/eyeballer/releases) and place it in the root of the repository. This is the only file needed for prediction.

## Predicting Labels

NOTE: For best results, make sure you screenshot your websites in a native 1.6x aspect ratio. IE: 1440x900. Eyeballer will scale the image down automatically to the right size for you, but if it's the wrong aspect ratio then it will squish in a way that will affect prediction performance.

To eyeball a single screenshot:

```
eyeballer.py predict YOUR_FILE.png
```

Or a whole directory of files:

```
eyeballer.py predict PATH_TO/YOUR_FILES/
```

For multi-file runs, Eyeballer writes:
- `results.html` — browsable visual report
- `results.csv` — machine-readable output (default)

A custom weights file can be specified with the global `--weights` flag:

```
eyeballer.py --weights YOUR_WEIGHTS.h5 predict YOUR_FILE.png
```

### Output format

By default results are written to `results.csv`. Pass `--format json` to get `results.json` instead, which contains the raw confidence scores for each label and pipelines cleanly into tools like `jq`, Burp, or Caido:

```
eyeballer.py predict PATH_TO/YOUR_FILES/ --format json
```

Example `results.json` entry:

```json
{
  "filename": "screenshots/admin.example.com.png",
  "custom404": 0.021,
  "login": 0.893,
  "webapp": 0.721,
  "oldlooking": 0.043,
  "parked": 0.009
}
```

Extract high-confidence login pages with `jq`:

```
jq '[.[] | select(.login > 0.7)] | map(.filename)' results.json
```

### Thresholds

The default confidence threshold for all labels is `0.5`. Override it globally:

```
eyeballer.py predict YOUR_FILES/ --threshold 0.6
```

Or tune individual labels independently with `--thresholds`, leaving the rest at the `--threshold` default. This is useful when you want higher recall on valuable targets (login pages) and higher precision on noise (parked domains):

```
eyeballer.py predict YOUR_FILES/ --thresholds login=0.3,parked=0.8
```

Valid label names: `custom404`, `login`, `webapp`, `oldlooking`, `parked`.

## Performance

Eyeballer's performance is measured against an evaluation dataset, which is 20% of the overall screenshots chosen at random. Since these screenshots are never used in training, they can be an effective way to see how well the model is performing. Here are the latest results:

| Overall Binary Accuracy | 93.52% |
| ------ |:-----:|
| **All-or-Nothing Accuracy** | **76.09%** |

**Overall Binary Accuracy** is probably what you think of as the model's "accuracy". It's the chance, given any single label, that it is correct.

**All-or-Nothing Accuracy** is more strict. For this, we consider all of an image's labels and consider it a failure if ANY label is wrong. This accuracy rating is the chance that the model correctly predicts all labels for any given image.

| Label | Precision | Recall |
| ------ | ------ |:-----:|
| Custom 404 | 80.20% | 91.01% |
| Login Page | 86.41% | 88.47% |
| Webapp | 95.32% | 96.83% |
| Old Looking | 91.70% | 62.20% |
| Parked Domain | 70.99% | 66.43% |

For a detailed explanation on [Precision vs Recall, check out Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall).

## Training
To train a new model, run:
```
eyeballer.py train
```

You'll want a machine with a good GPU for this to run in a reasonable amount of time. Setting that up is outside the scope of this readme, however.

This will output a new model file (weights.h5 by default).

## Evaluation

You just trained a new model, cool! Let's see how well it performs against some images it's never seen before, across a variety of metrics:

```
eyeballer.py --weights YOUR_WEIGHTS.h5 evaluate
```

The output will describe the model's accuracy in both recall and precision for each of the program's labels. (Including "none of the above" as a pseudo-label)
