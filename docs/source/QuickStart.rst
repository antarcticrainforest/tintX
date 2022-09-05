Quick usage reference
---------------------


Below you can find a quick guide outlining the usage of the ``tintX`` library

The tintX user interface
+++++++++++++++++++++++++

The :class:`RunDirectory` class serves as the main user interface to interact
with the tracking algorithm. To be able to track cells tint needs information
on the datasets. This datasets are usually saved to netCDF or grib files
or xarray ``Datasets``. To make use of this interface an instance of the
:class:`RunDirectory` class has to be created. This can be done in multiple
ways:

Using already opened to datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tintx import RunDirectory

    run_dir = RunDirectory(existing_xarray_dataset,
                           "variable_name",
                           x_coord="long_name",
                           y_coord="lat_name",
                           time_coord="time_name"
    )

Using data saved to files
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tintx import RunDirectory

    input_files = "/path/to/input_files/*.nc"
    run_dir = RunDirectory.from_files(input_files,
                                      "variable_name"
                                      start="2020-01-01T00:00",
                                      end="2020-12-31T12:50"
    )


Using previously tracked data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tintx import RunDirectory
    run_dir = RunDirectory.from_dataframe("output.hdf5)


Methods and properties
++++++++++++++++++++++
The following collection gives an overview of the usage of the created
:class:`RunDirectory` object which is referred as ``run_dir``:


Applying the tracking algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    num_cells = run_dir.get_tracks(min_size=2, field_thresh=1)

.. seealso::

    :py:mod:`tintx.config`

Accessing the cell tracks
~~~~~~~~~~~~~~~~~~~~~~~~~
Cell tracks are stored in a :py:mod:`pandas.DataFrame`

.. code-block:: python

    run_dir.tracks

Saving tracked cells to file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    num_cells = run_dir.get_tracks(min_size=2, field_thresh=1)
    run_dir.save_tracks("output.hdf5")

.. seealso::

    :class:`tintx.RunDirectory.from_dataframe`

Retrieving tuning parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tintx import RunDirectory
    run_dir = RunDirectory.from_dataframe("output.hdf5)
    parameters = run_dir.get_parameters()

.. seealso::

    :class:`tintx.RunDirectory.from_dataframe`
    :func:`tintx.config.get`
    :func:`save_tracks`

Accessing the data and metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :py:mod:`xarray.Dataset` holding the data that is tracked.

.. code-block:: python

    run_dir.data

- :py:mod:`xarray.DataArray` holding the information of the
  longitude/latitude/time coordinates.

.. code-block:: python

    run_dir.lon
    run_dir.lat
    run_dir.time

- Getting the first and last time step that is considered:

.. code-block:: python

    run_dir.start
    run_dir.end


- Getting the variable name of the field that is tracked:

.. code-block:: python

    run_dir.var_name

Visualising the tracked data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Plotting cell tracks:

.. code-block:: python

    ax = run.plot_trajectories(thresh=2, plot_style={"ms":25, "lw":1})


- Creating an animation of the tracked tracked cells:

.. code-block:: python

    anim = run.animate(vmax=3, fps=2, plot_style={"res": "10m", "lw":1})


.. seealso::

   Module :py:mod:`xarray`
        How to work with `xarray <https://docs.xarray.dev/en/stable/user-guide/index.html>`_
        datasets.
   Module :py:mod:`pandas`
        How to work with `pandas DataFrames <https://pandas.pydata.org/docs/user_guide/index.html>`_
   Module :py:mod:`cartopy`
        How to visualise geo spatial data with `cartopy <https://scitools.org.uk/cartopy/docs/latest>`_
   Class :py:mod:`matplotlib.animation.FuncAnimation`
        How to make use of the ojbect created by `FuncAnimation <https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_
