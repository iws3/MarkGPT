# Module 02 — Python & Mathematics Essentials
## Days 7–12 | Beginner–Intermediate

---

## Module Overview

Now that you understand what language models are, let's build the mathematical and programming foundations you'll need. This module doesn't assume you know Python or linear algebra — only curiosity and willingness to work through examples.

By the end of Module 02, you will:
- Write vectorized code in NumPy
- Manipulate data with Pandas
- Plot and visualize loss curves
- Understand matrix operations, calculus, and probability
- Have a solid foundation for the neural networks in Module 03
- Relate theory to MarkGPT architecture
- Complete hands-on exercises

## Structure

```
lessons/       - Conceptual explanations with code examples
exercises/     - Practical implementation exercises
projects/      - Larger projects (optional)
resources/     - Additional readings and links
```

## Time Estimate

- Lessons: 4-6 hours
- Exercises: 4-6 hours
- **Total: 8-12 hours per module**

## Key Concepts

[See lesson files for detailed content]

## Completion Checklist

- [ ] Read all lessons (L*_*.md files)
- [ ] Complete all exercises (day*_*.md files)
- [ ] Pass the module quiz (if provided)
- [ ] Understand connections to MarkGPT

## Resources

- Lesson references contain links to papers and tutorials
- http://markgpt-docs.com (forthcoming)
- GitHub discussions: https://github.com/yourusername/MarkGPT-LLM-Curriculum/discussions

## Next Module

See ../module-0$((i+1))/README.md for the next module.
## Python Essentials for Machine Learning

### Python Fundamentals Review
Python is the dominant language in ML due to its simplicity, readability, and ecosystem.

**Core Concepts**
- Variables: Store data in memory
- Data types: int, float, str, bool, list, dict, tuple, set
- Operators: Arithmetic, comparison, logical, membership
- Control flow: if/elif/else, loops (for, while)
- Functions: Reusable code blocks with parameters and returns
- Classes: Object-oriented programming foundation

### Data Structures

**Lists**
- Ordered, mutable collections
- Index: 0-based access
- Methods: append(), extend(), insert(), remove(), pop()
- List comprehension: Concise creation [x*2 for x in range(10)]

**Dictionaries**
- Key-value pairs, unordered (Python 3.7+ ordered)
- Efficient lookup by key
- Methods: keys(), values(), items(), get(), pop()
- Use case: Storing mapped data

**Tuples**
- Ordered, immutable sequences
- Hashable: Can be dict keys
- Unpacking: a, b, c = (1, 2, 3)
- Use case: Fixed collections, function returns

**Sets**
- Unordered, unique elements
- Operations: union, intersection, difference
- Use case: Removing duplicates, membership testing

### Functions and Scope

**Function Definition**
```python
def function_name(arg1, arg2=default_value, *args, **kwargs):
    '''Docstring explaining function'''
    # Function body
    return result
```

**Scope Rules (LEGB)**
- Local: Inside function
- Enclosing: In outer function
- Global: Module level
- Built-in: Python built-ins

**Lambda Functions**
- Anonymous functions: lambda x: x**2
- Use with map(), filter(), sorted()
- Avoid complex logic

### Error Handling

**Try-Except Pattern**
```python
try:
    # Code that might raise exception
except SpecificError as e:
    # Handle specific error
except Exception as e:
    # Catch all remaining
finally:
    # Always execute (cleanup)
```

**Common Exceptions**
- ValueError: Invalid value
- TypeError: Wrong type
- IndexError: Invalid index
- KeyError: Missing dictionary key
- ZeroDivisionError: Division by zero

### Modules and Packages

**Importing**
- import numpy: Full module namespace
- from numpy import array: Specific items
- from numpy import * : All items (avoid, causes conflicts)
- import numpy as np: Aliasing

**Creating Modules**
- File with .py extension is module
- Packages: Folders with __init__.py
- Relative imports: from . import sibling
- Absolute imports: from package.module import item

