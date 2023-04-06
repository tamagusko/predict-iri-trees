# Flexible Pavements Performance Prediction Using Machine Learning: Supervised Learning with Tree-Based Algorithms

This paper presents the application of supervised machine learning  tree-based algorithms to predict the performance of flexible pavements,  utilizing data from 55 experimental sections from the Long-Term Pavement Performance (LTPP) program. The International Roughness Index (IRI) was adopted as a pavement performance indicator. Decision Tree, Random  Forest, and XGBoost algorithms were employed in this study. Ultimately,  the best-trained model was XGBoost, achieving an R-squared of 0.98 and  an RMSE of 0.08 for the test sample.

Paper presented at the World Conference on Transport Research - WCTR 2023 Montreal 17-21 July 2023.

## Citation:

Tiago Tamagusko and Adelino Ferreira (2023). **Pavement Performance Prediction using Machine Learning: Supervised Learning with Tree-Based Algorithms**. World Conference on Transport Research - WCTR 2023.

<!-- [DOI:DOI](https://doi.org/doi) -->

```bibtex
@article{Tamagusko-Ferreira2023-predict-iri-tree,
  author = Tiago Tamagusko, Adelino Ferreira,
  title = "Pavement Performance Prediction using Machine Learning: Supervised Learning with Tree-Based Algorithms",
  journal = {World Conference on Transport Research - WCTR 2023},
  year = 2023,
  address   = "Canada, Montreal"
}
```

----

Please direct issues, bug reports and pull requests to this GitHub page. To contact me directly, send email to tamagusko@gmail.com.

-- Tiago

## How to use

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the Jupyter Notebook: `jupyter notebook main.ipynb`

## License

[CC-BY-NC-ND-4.0](LICENSE) (c) 2023, [Tiago Tamagusko](https://github.com/tamagusko).
