#!/usr/bin/env python3
# -*- coding: utf8 -*-

# pylint: disable=C0111  # docstrings are always outdated and wrong
# pylint: disable=W0511  # todo is encouraged
# pylint: disable=C0301  # line too long
# pylint: disable=R0902  # too many instance attributes
# pylint: disable=C0302  # too many lines in module
# pylint: disable=C0103  # single letter var names, func name too descriptive
# pylint: disable=R0911  # too many return statements
# pylint: disable=R0912  # too many branches
# pylint: disable=R0915  # too many statements
# pylint: disable=R0913  # too many arguments
# pylint: disable=R1702  # too many nested blocks
# pylint: disable=R0914  # too many local variables
# pylint: disable=R0903  # too few public methods
# pylint: disable=E1101  # no member for base
# pylint: disable=W0201  # attribute defined outside __init__
# pylint: disable=R0916  # Too many boolean expressions in if statement


import csv
import heapq
import logging
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path
from typing import ByteString
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence

import click
from enumerate_input import enumerate_input
#from with_sshfs import sshfs
#from with_chdir import chdir
from retry_on_exception import retry_on_exception

csv.field_size_limit(2**30)  # can't use sys.maxsize because of Windows error


class CsvSortError(Exception):
    pass


#from prettytable import PrettyTable
#output_table = PrettyTable()


def eprint(*args, **kwargs):
    if 'file' in kwargs.keys():
        kwargs.pop('file')
    print(*args, file=sys.stderr, **kwargs)


try:
    from icecream import ic  # https://github.com/gruns/icecream
except ImportError:
    ic = eprint


# import pdb; pdb.set_trace()
# #set_trace(term_size=(80, 24))
# from pudb import set_trace; set_trace(paused=False)


def csvsort(*,
            input_file: Path,
            output_file: Path,
            columns,
            verbose: bool,
            debug: bool,
            max_size=100,
            has_header=True,
            delimiter=',',
            show_progress=False,
            parallel=True,
            quoting=csv.QUOTE_MINIMAL,
            encoding=None,
            numeric_column=False,
            csv_reader=None,
            ):
    """Sort the CSV file on disk rather than in memory.

    The merge sort algorithm is used to break the file into smaller sub files

    Args:
        input_file: the CSV filename to sort.
        columns: a list of columns to sort on (can be 0 based indices or header keys).
        output_file: filename for sorted file.
        max_size: the maximum size (in MB) of CSV file to load in memory at once.
        has_header: whether the CSV contains a header to keep separated from sorting.
        delimiter: character used to separate fields, default ','.
        show_progress: A flag whether or not to show progress.
        quoting: How much quoting is needed in the final CSV file.  Default is csv.QUOTE_MINIMAL.
        encoding: The name of the encoding to use when opening or writing the csv files. Default is None which uses the system default.
        numeric_column: If columns being used for sorting are all numeric and the desired output is to have the sorting be done numerically rather than string based. Default, False, does string-based sorting.
        csv_reader: a pre-loaded instance of `csv.reader`. This allows you to supply a compatible stream for use in sorting.
    """

    with open(input_file, newline='', encoding=encoding) as input_fp:
        reader = csv.reader(input_fp, delimiter=delimiter)
        if debug:
            ic(reader)
            #import IPython; IPython.embed()

        if has_header:
            header = next(reader)
        else:
            header = None

        columns = parse_columns(columns, header)

        filenames = csvsplit(reader, max_size)
        if show_progress:
            logging.info('Merging %d splits' % len(filenames))

        if parallel:
            concurrency = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=concurrency) as pool:
                map_args = [(filename, columns, numeric_column, encoding)
                            for filename in filenames]
                pool.starmap(memorysort, map_args)
        else:
            for filename in filenames:
                memorysort(filename, columns, numeric_column, encoding)

        sorted_filename = mergesort(filenames,
                                    columns,
                                    numeric_column,
                                    encoding=encoding)

        # XXX make more efficient by passing quoting, delimiter, and moving result
        # generate the final output file
        with open(output_file,
                  'w',
                  newline='',
                  encoding=encoding) as output_fp:
            writer = csv.writer(output_fp, delimiter=delimiter, quoting=quoting)
            if header:
                writer.writerow(header)
            with open(sorted_filename, newline='', encoding=encoding) as sorted_fp:
                for row in csv.reader(sorted_fp):
                    writer.writerow(row)

        os.remove(sorted_filename)


def parse_columns(columns, header):
    """check the provided column headers"""
    for i, column in enumerate(columns):
        if isinstance(column, int):
            if header:
                if column >= len(header):
                    raise CsvSortError(
                        'Column index is out of range: "{}"'.format(column))
        else:
            # find index of column from header
            if header is None:
                raise CsvSortError(
                    'CSV needs a header to find index of this column name:' +
                    ' "{}"'.format(column))
            else:
                if column in header:
                    columns[i] = header.index(column)
                else:
                    raise CsvSortError(
                        'Column name is not in header: "{}"'.format(column))
    return columns


def csvsplit(reader, max_size):
    """Split into smaller CSV files of maximum size and return the filenames."""
    max_size = max_size * 1024 * 1024  # convert to bytes
    writer = None
    current_size = 0
    split_filenames = []

    # break CSV file into smaller merge files
    for row in reader:
        if writer is None:
            ntf = tempfile.NamedTemporaryFile(delete=False, mode='w')
            writer = csv.writer(ntf)
            split_filenames.append(ntf.name)

        writer.writerow(row)
        current_size += sys.getsizeof(row)
        if current_size > max_size:
            writer = None
            current_size = 0
    return split_filenames


def memorysort(filename, columns, numeric_column, encoding=None):
    """Sort this CSV file in memory on the given columns"""
    with open(filename, newline='', encoding=encoding) as input_fp:
        rows = [row for row in csv.reader(input_fp) if row]

    rows.sort(key=lambda row: get_key(row, columns, numeric_column))
    with open(filename, 'w', newline='', encoding=encoding) as output_fp:
        writer = csv.writer(output_fp)
        for row in rows:
            writer.writerow(row)


def get_key(row, columns, numeric_column):
    """Get sort key for this row"""
    if (numeric_column):
        return [float(row[column]) for column in columns]
    else:
        return [row[column] for column in columns]


def decorated_csv(filename, columns, numeric_column, encoding=None):
    """Iterator to sort CSV rows
    """
    with open(filename, newline='', encoding=encoding) as fp:
        for row in csv.reader(fp):
            yield get_key(row, columns, numeric_column), row


def mergesort(sorted_filenames,
              columns,
              numeric_column,
              nway=2,
              encoding=None):
    """Merge these 2 sorted csv files into a single output file
    """
    merge_n = 0
    while len(sorted_filenames) > 1:
        merge_filenames, sorted_filenames = \
            sorted_filenames[:nway], sorted_filenames[nway:]

        with tempfile.NamedTemporaryFile(delete=False, mode='w') as output_fp:
            writer = csv.writer(output_fp)
            merge_n += 1
            for _, row in heapq.merge(*[
                    decorated_csv(filename, columns, numeric_column, encoding)
                    for filename in merge_filenames
            ]):
                writer.writerow(row)

            sorted_filenames.append(output_fp.name)

        for filename in merge_filenames:
            os.remove(filename)
    return sorted_filenames[0]


@click.command()
@click.argument("input_file",
                type=click.Path(exists=True,
                                dir_okay=False,
                                file_okay=True,
                                path_type=str,
                                allow_dash=False,),
                nargs=1,
                required=True,)
@click.argument("output_file",
                type=click.Path(exists=False,
                                dir_okay=False,
                                file_okay=True,
                                path_type=str,
                                allow_dash=True,),
                nargs=1,
                required=False,)
@click.option('--column', '--field', 'columns', type=int, multiple=True, help='column(s) of CSV to sort on')
@click.option('--max-size', type=int, default=100, help='maximum size of each split CSV file in MB (default 100)')
@click.option('--no-header', help='set CSV file has no header')
@click.option('--delimiter', type=str, default=',', help='set CSV delimiter (default ",")')
@click.option('--encoding', type=str, help='character encoding (eg utf-8) to use when reading/writing files (default uses system default)')
@click.option('--verbose', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option("--printn", is_flag=True)
@click.pass_context
def cli(ctx,
        input_file: str,
        output_file: str,
        columns,
        max_size: int,
        no_header: bool,
        delimiter: str,
        encoding: str,
        verbose: bool,
        debug: bool,
        printn: bool,
        ):

    has_header = not no_header

    if not output_file:
        output_file = '/dev/stdout'

    input_file = Path(input_file)
    output_file = Path(output_file)

    csvsort(input_file=input_file,
            output_file=output_file,
            columns=columns,
            max_size=max_size,
            has_header=has_header,
            delimiter=delimiter,
            encoding=encoding,
            verbose=verbose,
            debug=debug,
            )

#        if ipython:
#            import IPython; IPython.embed()

