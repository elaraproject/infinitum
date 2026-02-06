# Infinitum

This is a Desmos-like webapp that serves as a frontend to Elara symbolic and SymPy, allowing using the library without any coding experience. Includes facilities for computing derivatives and integrals (both symbolically and numerically) as well as solving differential equations.

> Currently the app is MIT-licensed, but if all contributors are willing, I would be more than happy to turn it into public-domain licensed software.

## Why the name?

The name _infinitum_ comes from Latin and means "infinity". The name was chosen because a large number of the (planned) functionalities of the app are calculus-based, such as solving differential equations. Historically, calculus was known as _infinitesimal calculus_, hence the origin of the name _infinitum_ for the app.

> **Note:** The name right now is TBD, we may keep it as this or switch to another name.

## Usage

Assuming you have [Streamlit](https://streamlit.io/) installed,

First clone the repository on to your machine:

```bash
git clone https://codeberg.org/elaraproject/elara-symbolic-ui.git
```

Then, from the repo you have cloned, just run:

```bash
python -m streamlit src/main.py
```

## Future Features

- Create an interface capable of processing and solving differential equations input as LaTeX by users
- Add graphs of the differential equations so that the user can visualize the equation they have input

## Contributors
[Jacky Song](https://codeberg.org/songtech-0912)
[Jacob Thomas](https://codeberg.org/NHWXCodeberg)
[David Yang](https://codeberg.org/David_Y)
