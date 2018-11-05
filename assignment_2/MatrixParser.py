import string
from Matrix import Matrix


class MatrixParser:
    def __init__(self):
        # whether parsing has completed, or we need to expect another line
        self.complete = True
        self.row_sep = ';'
        self.col_sep = ','

    def parse(self, s: str):
        """
        Parse string to matrix
        Expected format example: [[1, 2]; [3,2]; [3,4]]

        Note:
         If value has started, there should be terminating character after it.
         For example, `2]` if a correct value 2, `2 ` is not.
        :return: Matrix object
        """
        self.complete = False
        parsed = []
        matrix_start = False
        row_start = False
        row_separated = False
        row_index = 0
        value = None
        for i, c in enumerate(s):
            if c in string.whitespace:
                if value is not None:
                    self._char_err(s, i)
                continue
            if c == '[':
                if not matrix_start:
                    matrix_start = True
                elif not row_start:
                    row_start = True
                    parsed.append([])
                else:
                    self._char_err(s, i)
            elif c == ']':
                row_separated = False
                if value:
                    self._val_to_lst(parsed[row_index], value)
                    value = None
                if row_start:
                    row_start = False
                elif matrix_start:
                    matrix_start = False
                    self.complete = True
                else:
                    self._char_err(s, i)
            elif c in "0123456789":
                if value is None:
                    value = c
                else:
                    value += c
            elif c == '.' and value is not None:
                value += c
            elif c == self.col_sep and row_start:
                if value:
                    self._val_to_lst(parsed[row_index], value)
                    value = None
                else:
                    self._char_err(s, i)
                continue
            elif c == self.row_sep and matrix_start:
                if row_start:
                    self._char_err(s, i)
                if not row_separated:
                    row_separated = True
                else:
                    self._char_err(s, i)
                row_index += 1
                row_start = False
                continue
            else:
                self._char_err(s, i)
        if not self.complete:
            raise ValueError("Matrix is not ended properly")
        return Matrix(parsed)

    def _char_err(self, s, i):
        """
        Output Syntax error and pointer (literally) to invalid character
        :param s:
        :param i:
        :return:
        """
        start_i = max(i - 3, 0)
        end_i = min(len(s), i + 3)
        err_str = f"Bad char: `{s[i]}` (pos: {i}) in {s[start_i:end_i]}"
        filler = " " * (len(err_str) + 10)
        pos_str = "\n" + filler + "↑" + " " * 2
        raise SyntaxError(err_str + pos_str)

    def _val_to_lst(self, lst: list, val: str):
        """
        Convert value from string to it's correct type
        :param lst:
        :param val:
        :return:
        """
        if '.' in val:
            lst.append(float(val))
        else:
            lst.append(int(val))
