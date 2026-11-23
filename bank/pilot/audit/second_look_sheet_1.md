# Label check, second look (9 items)

Your first verdict on these items differs from our label. Our label comes from running the program on EvalPlus's tests; they are shown below. For each item, write in `second_look_answers.csv`:

- `label_error`: **yes** if our label is wrong for the problem as written, **no** if it is right.
- `reason`: one short sentence. It is needed for every item.

The question is the same as before: a program is "correct" if it gives the right result for every input the problem allows, edge cases included, and "wrong" if some allowed input gives a wrong result, an exception or an endless run. Style and efficiency do not matter. So decide whether the failing tests are inputs the problem allows, and whether the expected results are what the problem asks for. Where all tests pass, decide whether you can name an allowed input that the program gets wrong.

Decide each item on its own. Changing your first verdict and keeping it are both fine. Work alone and without AI tools. You may run the programs.

## a014

**Our label:** wrong. **Your first verdict:** correct.

**Problem**

```
from typing import List, Any


def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """
```

**Candidate**

```python
from typing import List, Any

def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """
    return [v for v in values if isinstance(v, int)]

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 953, 166 failed.
- The first three failures (the input is the list of arguments):

```
input:    [[True, False, None, 0, -10, 'test', [], {}, 3.14]]
expected: [0, -10]
program:  [True, False, 0, -10]

input:    [['apple', 2.71828, None, False, 'watermelon', 42]]
expected: [42]
program:  [False, 42]

input:    [[True, False, None, 0, 3.253739830287621, -10, 'test', [], {}, 3.14]]
expected: [0, -10]
program:  [True, False, 0, -10]
```

## a042

**Our label:** wrong. **Your first verdict:** correct.

**Problem**

```
"""
Write a function which given a matrix represented as a list of lists returns the max of the n'th column.
assert max_of_nth([[5, 6, 7], [1, 3, 5], [8, 9, 19]], 2) == 19
"""
```

**Candidate**

```python
def max_of_nth(matrix, n):
    """
    This function takes a 2D list (matrix) and an integer n as input.
    It returns the maximum value in the n-th column of the matrix.
    
    :param matrix: list of lists representing a 2D matrix
    :param n: column index (0-based)
    :return: maximum value in the n-th column
    """
    # Check if the matrix is not empty and the column index is within bounds
    if not matrix or n < 0 or n >= len(matrix[0]):
        raise ValueError("Invalid input: matrix is empty or column index out of range")

    # Initialize max_value with the first column value, ensuring it's within bounds
    max_value = matrix[0][n]
    
    # Iterate over each row in the matrix
    for row in matrix:
        if row[n] > max_value:
            max_value = row[n]
    
    return max_value

import unittest

class TestMaxOfNth(unittest.TestCase):
    def test_max_of_nth(self):
        self.assertEqual(max_of_nth([[5, 6, 7], [1, 3, 5], [8, 9, 19]], 2), 19)
        self.assertEqual(max_of_nth([[10, 20, 30], [40, 50, 60], [70, 80, 90]], 1), 80)
        self.assertEqual(max_of_nth([[2, 4, 6], [8, 10, 12], [14, 16, 18]], 0), 18)
        self.assertRaises(ValueError, max_of_nth, [], 2)
        self.assertRaises(ValueError, max_of_nth, [[1, 2, 3]], -1)

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 101, 23 failed.
- The first three failures (the input is the list of arguments):

```
input:    [[[-5, -2, -10], [-5, -2, -10], [-8, -3, -7], [-6, -9, -4]], -1]
expected: -4
program:  error: ValueError: Invalid input: matrix is empty or column index out of range

input:    [[[-5, -2, -10], [-6, -9, -4], [-5, -2, -10]], -1]
expected: -4
program:  error: ValueError: Invalid input: matrix is empty or column index out of range

input:    [[[-5, -2, -10], [-8, -3, -7], [-6, -9, -4]], -2]
expected: -2
program:  error: ValueError: Invalid input: matrix is empty or column index out of range
```

## a046

**Our label:** correct. **Your first verdict:** wrong.

**Problem**

```
"""
Write a function to find the minimum value in a given heterogeneous list.
assert min_val(['Python', 3, 2, 4, 5, 'version'])==2
"""
```

**Candidate**

```python
def min_val(lst):
    return min(i for i in lst if isinstance(i, int))

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 104, 0 failed.

## a076

**Our label:** correct. **Your first verdict:** wrong.

**Problem**

```
"""
Write a function that checks if a strings contains 'z', except at the start and end of the word.
assert text_match_wordz_middle("pythonzabc.")==True
"""
```

**Candidate**

```python
def text_match_wordz_middle(text):
    words = text.split()
    for word in words:
        if 'z' in word[1:-1]:
            return True
    return False

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 103, 0 failed.

## a097

**Our label:** wrong. **Your first verdict:** correct.

**Problem**

```
"""
Write a function which given a matrix represented as a list of lists returns the max of the n'th column.
assert max_of_nth([[5, 6, 7], [1, 3, 5], [8, 9, 19]], 2) == 19
"""
```

**Candidate**

```python
def max_of_nth(matrix, n):
    """
    Return the maximum value in the n'th column of a given matrix.
    
    :param matrix: List of lists representing a matrix
    :param n: Index of the column for which the maximum is to be found
    :return: Maximum value in the n'th column
    """
    if not matrix or n < 0:
        raise ValueError("Invalid matrix or column index")
    
    nth_column = [row[n] for row in matrix if n < len(row)]
    return max(nth_column)

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 101, 23 failed.
- The first three failures (the input is the list of arguments):

```
input:    [[[-5, -2, -10], [-5, -2, -10], [-8, -3, -7], [-6, -9, -4]], -1]
expected: -4
program:  error: ValueError: Invalid matrix or column index

input:    [[[-5, -2, -10], [-6, -9, -4], [-5, -2, -10]], -1]
expected: -4
program:  error: ValueError: Invalid matrix or column index

input:    [[[-5, -2, -10], [-8, -3, -7], [-6, -9, -4]], -2]
expected: -2
program:  error: ValueError: Invalid matrix or column index
```

## a113

**Our label:** wrong. **Your first verdict:** correct.

**Problem**

```
"""
Write a function to find the minimum value in a given heterogeneous list.
assert min_val(['Python', 3, 2, 4, 5, 'version'])==2
"""
```

**Candidate**

```python
def min_val(lst):
    """
    Function to find the minimum value in a heterogeneous list.

    Parameters:
    lst (list): A list containing various data types.

    Returns:
    int/float/str: The minimum value in the list. If the list is empty, returns None.
    """
    if not lst:
        return None
    
    # Using a generator expression to filter out non-numeric values
    numeric_values = [x for x in lst if isinstance(x, (int, float))]
    
    if numeric_values:
        return min(numeric_values)
    else:
        return min(lst, key=lambda x: float('-inf') if not isinstance(x, (int, float)) else x)

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 104, 8 failed.
- The first three failures (the input is the list of arguments):

```
input:    [['Python', 10, 5, 2.5, 2.5, 'version']]
expected: 5
program:  2.5

input:    [['Python', 10, 3, 2.5, 2.5, 'version']]
expected: 3
program:  2.5

input:    [['Python', 4, 10, 3, 2.5, 2.5, 'version']]
expected: 3
program:  2.5
```

## a142

**Our label:** wrong. **Your first verdict:** correct.

**Problem**

```
"""
Write a python function to check whether the count of divisors is even. 
assert count_divisors(10)
"""
```

**Candidate**

```python
def count_divisors(n: int) -> bool:
    """
    Returns True if the count of divisors of n is even, False otherwise.
    """
    count = 0
    for i in range(1, n + 1):
        if n % i == 0:
            count += 1
    return count % 2 == 0

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 113, 9 failed.
- The first three failures (the input is the list of arguments):

```
input:    [987654321]
expected: True
program:  no result after 30 s

input:    [987654320]
expected: True
program:  no result after 30 s

input:    [987654322]
expected: True
program:  no result after 30 s
```

## a177

**Our label:** correct. **Your first verdict:** wrong.

**Problem**

```
"""
Write a function that checks if a strings contains 'z', except at the start and end of the word.
assert text_match_wordz_middle("pythonzabc.")==True
"""
```

**Candidate**

```python
def text_match_wordz_middle(text):
    """
    Check if a string contains 'z' at positions other than the first and last.
    
    Args:
        text (str): Input string.

    Returns:
        bool: True if 'z' exists in the middle of the string, False otherwise.
    """
    # Trim whitespaces and check if the string is not empty
    text = text.strip()
    if not text:
        return False

    # Check if 'z' exists anywhere in the middle of the string, excluding first and last positions
    for i in range(1, len(text) - 1):
        if text[i] == 'z':
            return True
    return False

import unittest

class TestTextMatchWordZMiddle(unittest.TestCase):
    def test_text_match_wordz_middle(self):
        self.assertTrue(text_match_wordz_middle("pythonzabc."))
        self.assertTrue(text_match_wordz_middle("zinmiddle"))
        self.assertTrue(text_match_wordz_middle("z_middle_z"))
        self.assertFalse(text_match_wordz_middle("pz.z"))
        self.assertFalse(text_match_wordz_middle("zabc"))
        self.assertFalse(text_match_wordz_middle(""))
        self.assertFalse(text_match_wordz_middle("   "))

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 103, 0 failed.

## a181

**Our label:** wrong. **Your first verdict:** correct.

**Problem**

```
"""
Write a function to remove uppercase substrings from a given string.
assert remove_uppercase('cAstyoUrFavoRitETVshoWs') == 'cstyoravoitshos'
"""
```

**Candidate**

```python
def remove_uppercase(string):
    """
    Function to remove uppercase substrings from a given string.

    :param string: The input string.
    :return: The string with all uppercase substrings removed.
    """
    result = ""
    for char in string:
        if not char.isupper():
            result += char
    return result

```

**Tests**

- Original tests: 3, 0 failed. Extra tests: 103, 78 failed.
- The first three failures (the input is the list of arguments):

```
input:    ['ThiS%^%!s&a(mY)TesTStR%i*ng']
expected: 'hisamesting'
program:  'hi%^%!s&a(m)est%i*ng'

input:    ['ThiS%^%!s&a(mY)TsesTStR%i*ng']
expected: 'hisamsesting'
program:  'hi%^%!s&a(m)sest%i*ng'

input:    ['ThiS%^%!%s*ng']
expected: 'hisng'
program:  'hi%^%!%s*ng'
```
