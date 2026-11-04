# LaTeX Academic Paper Template

A professional LaTeX template for academic papers, featuring CMU and LTI branding and optimized for research publications.

## Overview

This template provides a clean, well-structured foundation for academic papers. It includes:

- Professional CMU and LTI logo integration
- Standardized section structure (Introduction, Related Work, Methodology, Experiments, Conclusion)
- Optimized typography and formatting
- Bibliography support with BibTeX
- Figure and table management
- Comprehensive commenting and guidance

## Features

- **Professional Branding**: Integrated CMU wordmark and LTI logos
- **Structured Layout**: Pre-organized sections with guidance comments
- **Typography**: Clean, readable fonts optimized for academic writing
- **Bibliography**: BibTeX integration for reference management
- **Figures & Tables**: Organized directory structure for assets
- **Customizable**: Easy to modify for different paper types and requirements

## Quick Start

1. **Clone or download** this template
2. **Edit the main content**:
   - Update `main.tex` with your paper title, authors, and abstract
   - Modify section files in `sec/` directory with your content
   - Add your references to `ref.bib`
3. **Add your assets**:
   - Place figures in `figs/` directory
   - Place tables in `tabs/` directory
4. **Compile**: Use your preferred LaTeX compiler (pdflatex, xelatex, etc.)

## File Structure

```
├── main.tex              # Main document file
├── pre.tex               # Preamble with packages and commands
├── ref.bib               # Bibliography file
├── sec/                  # Section files
│   ├── intro.tex         # Introduction
│   ├── related.tex       # Related Work
│   ├── method.tex        # Methodology
│   ├── experiment.tex    # Experiments
│   └── conclusion.tex    # Conclusion
├── figs/                 # Figures directory
├── tabs/                 # Tables directory
├── logo/                 # Logo directory
│   ├── cmu.png           # CMU logo
│   ├── lti.png           # LTI logo
│   ├── github.png        # GitHub logo
│   └── huggingface.png   # HuggingFace logo
├── LICENSE               # CC-BY-SA 4.0 License
└── README.md             # This file
```

## Customization

### Changing Logos
Replace logos in the `logo/` directory with your institution's logos. Update the logo references in `main.tex` if needed.

### Using GitHub/HuggingFace Logos
The template includes GitHub and HuggingFace logos. To use them with your links, uncomment the alternative logo version in `main.tex`:

```latex
% Alternative with logos (uncomment to use):
\href{https://github.com/your-username/your-repo}{\includegraphics[height=0.4cm]{logo/github.png} \textbf{your-username/your-repo}} ~ ~ ~ \href{https://huggingface.co/datasets/your-username/your-dataset}{\includegraphics[height=0.4cm]{logo/huggingface.png} \textbf{your-username/your-dataset}}
```

### Modifying Sections
- Add new sections by creating `.tex` files in the `sec/` directory
- Include them in `main.tex` using `\input{sec/newsection}`
- Remove unused sections by commenting out their `\input` lines

### Bibliography Style
The template uses a standard bibliography style. Modify the `\bibliographystyle{}` command in `main.tex` to change citation formats.

### Custom Commands
Add your custom LaTeX commands and macros to `pre.tex` for consistency across the document.

## Compilation

### Standard Compilation
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using Latexmk (Recommended)
```bash
latexmk -pdf main.tex
```

### Overleaf
Upload all files to Overleaf and set the main document to `main.tex`.

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- Required packages (automatically handled by most distributions):
  - Standard LaTeX packages
  - Graphics packages for logo inclusion
  - Bibliography packages

## Credits and Attribution

This template is derived from the LaTeX source of the paper:

**"DeepDatasetResearch: A Comprehensive Benchmark for Evaluating Agent Systems in Demand-Driven Dataset Discovery and Synthesis"**

**Original Authors:**
- Zihan Yang (Carnegie Mellon University)
- Yixuan Zhang (Carnegie Mellon University)  
- Zhiyu Zoey Chen (Carnegie Mellon University)
- Jingya Chen (Carnegie Mellon University)
- Ruoxi Sun (Carnegie Mellon University)
- Xingyao Wang (Carnegie Mellon University)
- Hao Peng (University of Illinois Urbana-Champaign)
- Heng Ji (University of Illinois Urbana-Champaign)
- Graham Neubig (Carnegie Mellon University)

**Original Paper:** arXiv:2508.06960

The original paper content has been removed and replaced with generic template guidance while preserving the professional structure and formatting. The CMU and LTI logos have been integrated to reflect the institutional branding.

## License

This template is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0)**.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit to the original authors and this template
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

See the [LICENSE](LICENSE) file for the complete license text.

## Contributing

Contributions to improve this template are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request with a clear description of changes

## Support

For issues or questions about this template:

1. Check the LaTeX documentation for compilation issues
2. Review the file structure and comments for usage guidance
3. Consult LaTeX community resources for advanced customization

## Acknowledgments

- Original paper authors for the foundational structure
- Carnegie Mellon University and Language Technologies Institute for institutional branding
- LaTeX community for the underlying typesetting system