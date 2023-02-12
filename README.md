<a name="readme-top"></a>


<!-- SOCIAL MEDIA LINKS -->
<a href="https://www.linkedin.com/in/marc-espinos-longa/">
  <img src="images/pngegg.png" width="25" height="25">
</a>
&nbsp
<a href="https://github.com/M-Espinos-Longa">
  <img src="images/25231.png" width="25" height="25">
</a>
&nbsp
<a href="https://marcespinoslonga.wordpress.com/">
  <img src="images/wordpress.png" width="25" height="25">
</a>
&nbsp
<a href="https://orcid.org/0000-0001-7916-9383">
  <img src="images/ORCID_iD.svg.png" width="25" height="25">
</a>

<!-- COVER -->
<p align="center">
  <div align="center">
    <a href="https://www.transformer-technology.com/news/us-news/1898-national-grid-uk-to-acquire-western-power-distribution-from-ppl.html">
      <img src="images/elec.jpg" alt="Logo" width="512" height="240">
    </a>
    <h1 align="center"><strong> UK Power Demand Forecast </strong></h1>
    <p>
      Authors: Marc Espinós Longa <br/>
      Date: 11 Feb 2023
    </p>
  </div>
</p>


<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong> Table of Contents </strong></summary>
  <ol>
    <li><a href="#abstract"> Abstract </a></li>
    <li>
      <a href="#requirements"> Requirements </a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#system-design"> System Design </a></li>
    <li>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABSTRACT -->
<a name="abstract"></a>
## Abstract
<p align="justify"> In recent years, electricity generation has been transitioning towards zero-emission (green) energy sources. These processes often rely on external and non-controllable variables (e.g., weather conditions), making electricity production erratic and intermittent. To avoid energy shortage, the system must have an alternative supply that responds to the required demand when necessary. As energy storage is only feasible on a small to medium scale, electricity distributors need to look ahead and plan energy supply based on future power demand. </p>

<p align="justify"> Based on historical data extracted every 30 min. between 2017 and 2022, we will train and test an artificial neural network (ANN) that can efficiently forecast (RMSE<sup>1</sup> <= 12.416) future power demand (MW<sup>2</sup>) in an interval of 30 minutes given 12 hours of previous input data. </p>

<font size="-1"><sup>1</sup> Root mean square error is often used as an accuracy measure in time-series forecasting <a href="#r1">[1-4]</a>. </br>
<sup>2</sup> Megawatts.</font>
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REQUIREMENTS -->
<a name="requirements"></a>
## Requirements
<p align="justify"> This project is built in Python 3.10.6 and Linux operating system. However, all the included libraries are compatible with macOS and Windows. You can easily check your current Python 3 version by running <code>python3 --version</code> command on your terminal.</p>
<p align="justify"> List of libraries:</p>
<ul>
  <li>Pandas: read dataset file.</li>
  <li>Pytorch: neural networks and related.</li>
  <li>Wandb: real-time performance tracking.</li>
  <li>Matplotlib: graph visualisation.</li>
  <li>Tqdm: computing process visualisation on terminal.</li>
  <li>Numpy: random seed generator for experiment reproducibility.</li>
</ul>


<!-- INSTALLATION -->
<a name="installation"></a>
### Installation
<p align="justify">For Python versions greater than 3.5.x, <code>pip</code> package tool is installed by default. Earlier Python versions require <code>pip</code> to be installed separately. It is good practice to update your package information from all configured sources before making any installation. You can do that simply by running the following command:</p>

<pre><code class="sh">sudo apt-get update</code></pre>

<p align="justify">If you have already Python 3 installed in your system, execute the command below to install pip3:</p>

<pre><code class="sh">sudo apt-get -y install python3-pip</code></pre>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SYSTEM DESIGN -->
## System Design
<div id="system-design"></div>
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.


<!-- FUNCTION APPROXIMATION -->
### Function Approximation
<div id="function-approximation"></div>

<!-- DATA -->
### Data
<div id="data"></div>

<!-- HYPERPARAMETERS -->
### Hyperparameters
<div id="hyperparameters"></div>

<!-- RESULTS -->
## Discussion
<div id="discussion"></div>

<!-- CONCLUSION AND FUTURE WORK -->
## Conclusion and Future Work
<div id="conclusion-and-future-work"></div>

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CITATION -->
## Citation


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Othneil Drew, README.md template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REFERENCES -->
## References
<div class="csl-entry", id="r1", align="justify">[1] Georga, E. I., Fotiadis, D. I., &#38; Tigas, S. K. (2018). Nonlinear Models of Glucose Concentration. <i>Personalized Predictive Modeling in Type 1 Diabetes</i>, 131–151. <a href="https://doi.org/10.1016/B978-0-12-804831-3.00006-6">https://doi.org/10.1016/B978-0-12-804831-3.00006-6</a></div>

<div class="csl-entry", align="justify">[2] Christie, D., &#38; Neill, S. P. (2022). Measuring and Observing the Ocean Renewable Energy Resource. <i>Comprehensive Renewable Energy</i>, 149–175. <a href="https://doi.org/10.1016/B978-0-12-819727-1.00083-2">https://doi.org/10.1016/B978-0-12-819727-1.00083-2</a></div>

<div class="csl-entry", align="justify">[3] Functional Networks. (2005). In <i>Mathematics in Science and Engineering</i> (Vol. 199, Issue C, pp. 169–232). Elsevier. <a href="https://doi.org/10.1016/S0076-5392(05)80012-8">https://doi.org/10.1016/S0076-5392(05)80012-8</a></div>

<div class="csl-entry", align="justify">[4] Tiwari, K., &#38; Young Chong, N. (2020). Informative Path Planning (IPP): Informative area coverage. <i>Multi-Robot Exploration for Environmental Monitoring</i>, 85–99. <a href="https://doi.org/10.1016/B978-0-12-817607-8.00021-6">https://doi.org/10.1016/B978-0-12-817607-8.00021-6</a></div>
</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: images/pngegg.png
[linkedin-url]: https://www.linkedin.com/in/marc-espinos-longa/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
