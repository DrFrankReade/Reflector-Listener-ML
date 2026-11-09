"""Browser-style reflector favorites and nested folders."""

from dataclasses import asdict, dataclass, replace
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import weakref

from reflector_listener.protocols import parse_target
from reflector_listener.session import ConnectionState
from .tooltip import ToolTip


def folder_path(value):
    parts = tuple(part.strip() for part in value.split("/")) if value.strip() else ()
    if len(parts) > 8 or any(not part or len(part) > 80 or not part.isprintable() for part in parts):
        raise ValueError("Enter up to eight folder names separated by /.")
    return parts


@dataclass(frozen=True)
class Favorite:
    name: str
    reflector: str
    module: str
    host: str = ""
    folder: tuple[str, ...] = ()

    @classmethod
    def create(cls, name, reflector, module, host, folder):
        name = name.strip()
        if not name or len(name) > 80 or not name.isprintable():
            raise ValueError("Enter a favorite name of up to 80 characters.")
        target = parse_target(reflector, module)
        host = host.strip()
        if len(host) > 253 or any(char.isspace() or not char.isprintable() for char in host):
            raise ValueError("Enter a hostname or IP address, or leave Host blank.")
        return cls(name, target.name, target.module, host, folder_path(folder))


class _FavoriteDialog(simpledialog.Dialog):
    def __init__(self, parent, title, favorite, folders):
        self.favorite, self.folders = favorite, folders
        super().__init__(parent, title)

    def body(self, master):
        self.entries = {}
        for row, (label, value) in enumerate((
                ("Name", self.favorite.name), ("Reflector", self.favorite.reflector),
                ("Module", self.favorite.module), ("Host override", self.favorite.host),
                ("Folder", "/".join(self.favorite.folder)))):
            ttk.Label(master, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            if label == "Folder":
                entry = ttk.Combobox(master, width=36, values=("", *self.folders))
            else:
                entry = ttk.Entry(master, width=38)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            self.entries[label] = entry
        ttk.Label(master, text="Leave Folder blank for the main menu. Use / for subfolders.").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=6)
        return self.entries["Name"]

    def validate(self):
        try:
            self.result = Favorite.create(*(entry.get() for entry in self.entries.values()))
        except ValueError as exc:
            messagebox.showerror("Check favorite", str(exc), parent=self)
            return False
        return True


class Favorites:
    def __init__(self, app, menu_bar, connection, saved):
        self.app = weakref.proxy(app)
        self.items = []
        self.folders = set()
        self.manager = None
        self._load(saved)
        self.menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Favorites", menu=self.menu)
        self.star = ttk.Button(connection, width=3, command=self.add_current)
        self.star.grid(row=1, column=1, padx=(0, 5), sticky="w")
        ToolTip(self.star, "Add or edit the current reflector and module in Favorites.")
        self._traces = [(variable, variable.trace_add("write", self._refresh_star))
                        for variable in (app.reflector_var, app.module_var)]
        self.star.bind("<Destroy>", self._remove_traces, add="+")
        self._refresh_menu()

    def _remove_traces(self, _event):
        for variable, token in self._traces:
            variable.trace_remove("write", token)
        self._traces.clear()

    def _load(self, saved):
        if not isinstance(saved, dict):
            return
        folders = saved.get("folders")
        for path in folders if isinstance(folders, list) else ():
            try:
                self._ensure_folder(folder_path(path))
            except (AttributeError, ValueError):
                continue
        items = saved.get("items")
        for item in items if isinstance(items, list) else ():
            try:
                favorite = Favorite.create(item["name"], item["reflector"], item["module"],
                                           item.get("host", ""), "/".join(item.get("folder", [])))
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            self.items.append(favorite)
            self._ensure_folder(favorite.folder)

    def saved(self):
        return {"folders": ["/".join(path) for path in sorted(self.folders)],
                "items": [asdict(item) for item in self.items]}

    def _ensure_folder(self, path):
        self.folders.update(path[:size] for size in range(1, len(path) + 1))

    def _current_index(self):
        name = self.app.reflector_var.get().strip().upper()
        module = self.app.module_var.get().strip().upper()
        return next((index for index, item in enumerate(self.items)
                     if (item.reflector, item.module) == (name, module)), None)

    def _refresh_star(self, *_):
        self.star.configure(text="☆" if self._current_index() is None else "★")

    def _refresh_menu(self):
        self.menu.delete(0, "end")
        for child in self.menu.winfo_children():
            child.destroy()
        self.menu.add_command(label="Add or edit current reflector…", command=self.add_current)
        self.menu.add_command(label="Manage Favorites…", command=self.manage)
        if self.items or self.folders:
            self.menu.add_separator()
        menus = {(): self.menu}
        for path in sorted(self.folders):
            parent = menus[path[:-1]]
            submenu = menus[path] = tk.Menu(parent, tearoff=False)
            parent.add_cascade(label=path[-1], menu=submenu)
        for favorite in self.items:
            menus[favorite.folder].add_command(label=favorite.name, command=lambda item=favorite: self._select(item))
        self._refresh_star()

    def _changed(self):
        self._refresh_menu()
        if self.manager is not None and self.manager.winfo_exists():
            self._refresh_tree()
        self.app._save_settings()

    def _select(self, favorite):
        if self.app.session.state in (ConnectionState.LINKED, ConnectionState.CONNECTING, ConnectionState.STOPPING):
            messagebox.showinfo("Favorites", "Disconnect before selecting a different reflector.", parent=self.app.root)
            return
        self.app.reflector_var.set(favorite.reflector)
        self.app.module_var.set(favorite.module)
        self.app._on_reflector_selected()
        self.app.host_var.set(favorite.host or self.app._host_by_name.get(favorite.reflector, ""))
        self.app._save_settings()

    def add_current(self):
        index = self._current_index()
        if index is not None:
            self._edit_item(index, self.app.root)
            return
        try:
            target = parse_target(self.app.reflector_var.get(), self.app.module_var.get())
        except ValueError as exc:
            messagebox.showerror("Cannot add favorite", str(exc), parent=self.app.root)
            return
        host = self.app.host_var.get().strip()
        if host == self.app._host_by_name.get(target.name):
            host = ""
        self._edit_item(None, self.app.root, Favorite(target.name, target.name, target.module, host))

    def _edit_item(self, index, parent, new=None):
        title = "Add Favorite" if index is None else "Edit Favorite"
        result = _FavoriteDialog(parent, title, new if index is None else self.items[index],
                                 ["/".join(path) for path in sorted(self.folders)]).result
        if result is None:
            return
        self._ensure_folder(result.folder)
        if index is None:
            self.items.append(result)
        else:
            self.items[index] = result
        self._changed()

    def manage(self):
        if self.manager is not None and self.manager.winfo_exists():
            self.manager.lift()
            return
        self.manager = window = tk.Toplevel(self.app.root)
        window.title("Manage Favorites")
        window.transient(self.app.root)
        body = ttk.Frame(window, padding=12)
        body.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(body, columns=("target",), selectmode="browse", height=13)
        self.tree.heading("#0", text="Name")
        self.tree.heading("target", text="Reflector / module")
        self.tree.column("#0", width=280)
        self.tree.column("target", width=140)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(body, command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        buttons = ttk.Frame(body)
        buttons.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for label, command in (("New Folder…", self._new_folder), ("Edit…", self._edit_selected),
                               ("Delete…", self._delete_selected)):
            ttk.Button(buttons, text=label, command=command).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Done", command=window.destroy).pack(side="right")
        self.tree.bind("<Double-1>", lambda event: self._edit_selected() if self.tree.identify_row(event.y) else None)
        window.bind("<Escape>", lambda _event: window.destroy())
        self._refresh_tree()

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        parents = {(): ""}
        self.nodes = {}
        for path in sorted(self.folders):
            node = self.tree.insert(parents[path[:-1]], "end", text=path[-1], open=True)
            parents[path] = node
            self.nodes[node] = path
        for index, item in enumerate(self.items):
            node = self.tree.insert(parents[item.folder], "end", text=item.name,
                                    values=(f"{item.reflector} {item.module}",))
            self.nodes[node] = index

    def _selected(self):
        return next((self.nodes[node] for node in self.tree.selection()), None)

    def _ask_folder(self, title, initial=""):
        value = simpledialog.askstring(title, "Folder path (use / for subfolders)",
                                       initialvalue=initial, parent=self.manager)
        if value is None:
            return None
        try:
            path = folder_path(value)
            if not path:
                raise ValueError("Enter a folder name.")
            return path
        except ValueError as exc:
            messagebox.showerror("Check folder", str(exc), parent=self.manager)
            return None

    def _new_folder(self):
        selected = self._selected()
        parent = selected if isinstance(selected, tuple) else self.items[selected].folder if selected is not None else ()
        path = self._ask_folder("New Folder", "/".join(parent) + "/" if parent else "")
        if path:
            self._ensure_folder(path)
            self._changed()

    def _edit_selected(self):
        selected = self._selected()
        if isinstance(selected, int):
            self._edit_item(selected, self.manager)
        elif isinstance(selected, tuple):
            new = self._ask_folder("Rename or Move Folder", "/".join(selected))
            if new is None or new == selected:
                return
            if new in self.folders or new[:len(selected)] == selected:
                messagebox.showerror("Cannot move folder", "Choose a new path outside the selected folder.", parent=self.manager)
                return

            def moved(path):
                return new + path[len(selected):] if path[:len(selected)] == selected else path

            paths = {moved(path) for path in self.folders}
            if any(len(path) > 8 for path in paths):
                messagebox.showerror("Cannot move folder", "Folder nesting is limited to eight levels.", parent=self.manager)
                return
            self.folders = paths
            self._ensure_folder(new)
            self.items = [replace(item, folder=moved(item.folder)) for item in self.items]
            self._changed()

    def _delete_selected(self):
        selected = self._selected()
        if selected is None:
            return
        label = self.items[selected].name if isinstance(selected, int) else "/".join(selected)
        prompt = f"Delete {label}?" if isinstance(selected, int) else f"Delete {label} and all its favorites and subfolders?"
        if not messagebox.askyesno("Delete Favorite", prompt, parent=self.manager):
            return
        if isinstance(selected, int):
            del self.items[selected]
        else:
            self.items = [item for item in self.items if item.folder[:len(selected)] != selected]
            self.folders = {path for path in self.folders if path[:len(selected)] != selected}
        self._changed()
