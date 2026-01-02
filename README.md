# LL(1) Parser in Python

This project is a **complete LL(1) parser implementation in Python**. It reads a context-free grammar from a file, transforms it into an LL(1)-compatible form, computes parsing sets, builds a predictive parsing table, and parses input strings while generating a **parse tree** and **step-by-step trace**.

The project is suitable for **Compiler Construction / Theory of Computation** coursework and demonstrations.

---

## ✨ Features

* Reads grammar from a text file
* Eliminates **left recursion** (direct & indirect)
* Applies **left factoring**
* Computes:

  * NULLABLE sets
  * FIRST sets
  * FOLLOW sets
* Constructs **LL(1) predictive parsing table**
* Detects **LL(1) conflicts** (FIRST/FIRST, FIRST/FOLLOW)
* Parses input strings using stack-based LL(1) parsing
* Displays:

  * Parsing table (formatted)
  * Parsing trace (step-by-step)
  * Parse tree (ASCII tree)

---

## 📁 Project Structure

```
.
├── code.py          # Main LL(1) parser implementation
├── grammar.txt      # Example grammar (expression grammar)
├── grammar2.txt     # Simple grammar example
└── README.md        # Project documentation
```

---

## 📜 Grammar File Format

Each grammar must follow these rules:

* Use `->` to separate LHS and RHS
* Separate symbols with **spaces**
* Use lowercase `e` for epsilon (ϵ)
* Use `|` for alternative productions
* The **first non-terminal** is treated as the start symbol
* Lines starting with `#` are treated as comments

### Example

```
S -> A B
A -> a | e
B -> b
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sherrytelli/LL1-parser.git
cd LL1-parser
```

---

## 🐍 Python Virtual Environment Setup (Recommended)

It is **strongly recommended** to run this project inside a virtual environment.

### 🔹 Create Virtual Environment

```bash
python3 -m venv venv
```

### 🔹 Activate Virtual Environment

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
```

---

## 📦 Install Dependencies

The project uses only one external library:

```bash
pip install tabulate
```

> 💡 Make sure your virtual environment is activated before installing dependencies.

---

## ▶️ Running the Parser

Use the following command:

```bash
python code.py grammar.txt
```

If no grammar file is provided, the program will display usage instructions and a sample grammar format.

---

## ⌨️ Input String Rules

After successful LL(1) verification, you can enter strings to parse.

Rules:

1. Separate each token with a **space**
2. Do **not** include `$` — it is added automatically
3. Type `exit` to quit

### Example Input

```
a + b * a
```

---

## 📊 Output Overview

During execution, the parser prints:

1. Original grammar
2. Grammar after left recursion removal
3. Grammar after left factoring
4. NULLABLE, FIRST, and FOLLOW sets
5. Predictive parsing table
6. LL(1) conflict report
7. Parsing trace (stack, input, action)
8. Final parse tree

---

## 🌳 Parse Tree Example

```
└── E
    ├── T
    │   └── F
    │       └── a
    ├── +
    └── F
        └── b
```

---

## ❌ Error Handling

* Grammar conflicts are reported clearly
* Parsing stops if grammar is **not LL(1)**
* Detailed error messages for mismatches and missing table entries

---

## 🧠 Educational Value

This project demonstrates:

* Grammar normalization techniques
* LL(1) parsing theory in practice
* Compiler front-end fundamentals
* Parse tree generation

Ideal for **students, educators, and compiler enthusiasts**.

---

## 📌 Notes

* Epsilon is represented by `e`
* Input tokens must exactly match grammar terminals
* The parser is **table-driven**, not recursive-descent

---

## 📜 License

No license has been added yet.

If you plan to share or publish this project, consider adding a license such as **MIT** or **Apache 2.0**.

---

## 👤 Author

**Sheheryar Salman**

---

Happy Parsing! 🚀
