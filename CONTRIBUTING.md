# Contributing to IKOB

First off, thank you for considering contributing to IKOB! It's much appreciated.

## Table of Contents

- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)


## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid 
duplicates. When you create a bug report, include as many details as 
possible.

**Great bug reports include:**
- A clear, descriptive title
- Steps to reproduce the behavior
- Expected behavior vs actual behavior
- Screenshots (if applicable)
- Environment details (project / python version etc.)

### 💡 Suggesting Features

Feature requests are welcome!

**Great feature requests include:**
- Clear problem statement
- Proposed solution
- Additional context
- Alternative solutions you've considered (if applicable)

### 📝 Improving Documentation

Documentation improvements are always welcome! This includes:
- Fixing typos
- Adding examples
- Clarifying confusing sections

## Development Setup

### Prerequisites

- python 3.13.1+
- Git

### Getting Started

 1. Fork the repository on GitHub. \
If you have push access to the IKOB repository you can skip this step, clone the original repo and work in there.

 2. Clone your fork locally
```bash
git clone https://github.com/YOUR_USERNAME/ikob.git
cd [project-name]
```
3. Add upstream remote.\
If you have push access, it's also not necessary to add an upstream remote, just use origin.
```bash
git remote add upstream https://github.com/Stichting-CROW/ikob.git
```
4. Create a branch for your changes
```bash
git checkout -b feature/your-feature-name
```
5. Setup development environment, see the [Development section](README.md#development) in the readme

6. push your changes locally
```bash
git push -u origin feature/your-feature-name
```

7. Create a pull request with 'compare across forks' on GitHub

## Pull Request Process

### Before Submitting

0. Know that your PR has a much better chance of being accepted if you **open an issue first** for discussion.

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

2. **format your code**
   ```bash
   ruff format .
   ```

3. **Run the full test suite** and ensure all tests pass:
   ```bash
   pytest .
   ```

### Submitting

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request against the `master` branch.

3. Wait for review.

### PR Checklist

- [ ] My code follows the project's [style guidelines](#style-guide)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas and focussing on _why_ the code is the way is it over _what_ the code doing. _What_ should ideally be clear from the code itself.
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass with my changes

## Style Guide

### Commit Messages

We follow [Conventional Commits](https://conventionalcommits.org/):

```
<type>(<optional scope>): <description> 

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Maintenance tasks

**Examples:**
```
fix(ui): Display correct units for parking costs
docs: Add contributing.md
```

### Code Style

- Use ruff

### Testing

- Bug fixes should include regression tests
- Tests should be deterministic (no flaky tests)

---

Thank you for contributing! 🎉
