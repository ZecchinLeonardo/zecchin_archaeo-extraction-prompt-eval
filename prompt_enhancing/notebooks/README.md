# To run notebooks with this project

Install on your user space `poetry-kernel`

```sh
pipx install poetry-kernel --include-deps
```

Then, run inside this project this

```sh
just install-notebookenv
```

Finally, you can run your jupyter instance (installed wherever you want)

```sh
jupyter notebook .
```
