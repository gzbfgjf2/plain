<!-- # plain -->

# Motivation

Inspired by miniGPT, and nanoGPT, this repo aims to

> be small, clean, interpretable and educational, as most of the currently
> available GPT model implementations can a bit sprawling.

In addition, this repo tries to embody a self-explanatory and modularised
nature, such that every function only consists of simple logic. So researchers
who wants to customise their models or try new ideas can just copy, paste and
edit the code.

Minimise source code size. Trade flexibility for simpleness of source code.

Researchers: change source code beginners: understand source code

All needed within a single file.

# Data interface

### data output

output all data needed a sequence do not handle shifting, because not all model
needs shifting, only autoregressive model needs shifts

to ids

### model output

Training step taking ids, output logits, loss

Evaluation step taking ids, output ids, loss


### conventions

When a function return prediction and label related data, return prediction
first
