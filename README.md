# Quoish's Library

This repository is a personal knowledge base, collecting notes, ideas, and drafts from daily study and projects.

It was originally created as an Obsidian vault and is now also used as the source for a GitHub Pages site, so that all notes can be read directly in the browser.

---

## 📖 Online site (GitHub Pages)

Once GitHub Pages is enabled in the repository settings, the site will be available at:

- https://sofaray5.github.io/Quoish-Library/

The site is generated with **GitHub Pages + Jekyll** using the built-in **minima** theme.

- `index.md` is the homepage of the site.
- Markdown files in the root folder (such as `欢迎.md`, `规范.md`, `C++数据结构.md`) are rendered as individual pages.
- Subfolders (`Matlab/`, `Python/`, `STM32/`, `数学物理/` etc.) act as topic collections.

---

## 📂 Repository structure (simplified)

Current top-level structure:

- `.obsidian/` – Obsidian configuration (plugins, settings, workspace, etc.)
- `Matlab/` – Notes and experiments related to Matlab.
- `Python/` – Python-related notes, scripts, and experiments.
- `STM32/` – Embedded development notes and code around STM32.
- `数学物理/` – Mathematical physics related material (course notes, derivations, exercises).
- `C++数据结构.md` – C++ data structure notes.
- `临时.md` – Temporary notes.
- `未命名.md`, `未命名 1.md` – Unnamed or scratch notes.
- `欢迎.md` – A “welcome” note for the vault.
- `规范.md` – Some conventions / standards for notes.
- `音频文件格式.md` – Notes about audio file formats.
- `README.md` – This file.
- `index.md` – Homepage content for GitHub Pages.
- `_config.yml` – Configuration file for Jekyll / GitHub Pages.

---

## ✍️ How to edit notes

You can edit notes in two ways:

1. **In Obsidian (local)**  
   - Clone the repo locally.  
   - Open the folder as an Obsidian vault.  
   - Edit Markdown files directly and commit + push to GitHub.

2. **Directly on GitHub**  
   - Open any `.md` file in the browser.
   - Click the “Edit this file” (pencil icon).
   - Commit changes to the `main` branch.

Any committed changes to `main` will automatically update the GitHub Pages site (after a short build).

---

## 🌐 Enabling GitHub Pages (one-time setup)

1. Go to the repository: `SofaRay5/Quoish-Library`.
2. Click **Settings**.
3. In the left sidebar, select **Pages**.
4. Under **Source**, choose:
   - **Deploy from a branch**
   - Branch: `main`
   - Folder: `/ (root)`
5. Save.

GitHub will build the site. After that, visit:

- https://sofaray5.github.io/Quoish-Library/

to browse the notes as a website.

---

## 🛠️ Jekyll configuration

The site uses the built-in **minima** theme and a minimal `_config.yml`:

- Site title, description, language, and GitHub username are configured.
- `url` and `baseurl` are set so that links work correctly under the repository path.

You can customize the theme, navigation, and layout later by editing `_config.yml` and adding more Jekyll pages/layouts if needed.

---

## 📌 Future ideas

- Add `README.md` inside each topic folder (`Matlab/`, `Python/`, `STM32/`, `数学物理/`) as a local index page.
- Rename scratch files like `未命名.md` to more meaningful names once their content stabilizes.
- Organize notes into a clearer hierarchy (by course, by topic, by project, etc.).
- Add tags, cross-links, or tables of contents to make it easier to navigate.

For now, the main goal is simple: **turn the existing vault into a browsable website with almost no extra work**.
