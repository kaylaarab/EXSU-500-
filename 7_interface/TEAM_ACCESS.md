# Team access: try the readmission demo with nothing to install

This page is for teammates (Larissa, Magalie, Kayla, Siham). The app runs in the cloud through GitHub Codespaces and is shown in your browser. It stays private: only the person who starts a codespace can see it.

## Steps
1. Sign in to GitHub and open <https://github.com/kaylaarab/EXSU-500-> (Siham's copy <https://github.com/skhaldoun/EXSU-500-> holds the same files).
2. Click the green **Code** button, open the **Codespaces** tab, then click **Create codespace on main**.
3. Wait 2 to 4 minutes the first time. The app opens in a new browser tab. If it does not, open the **Ports** tab at the bottom and click the globe next to port **8501**.
4. Use the **Risk calculator** tab: click an example patient, change any field, press calculate. The result is a risk percentage, a risk band and a chart of what pushed the risk up or down.
5. When you are done, stop the codespace: Code, Codespaces tab, the **...** menu next to your codespace, **Stop codespace**. This saves your free monthly hours.
6. Please do not make the port public. The interface is private until the whole project is finished and the team decides.

## Who can do this
- You need **Write** access to the repository (Kayla: Settings, Collaborators). Read-only collaborators do not get the Codespaces option.
- Each person's codespace uses their own free GitHub Codespaces hours.

## If something fails
- Open the **Terminal** tab in the codespace and run:
  ```
  python -m pip install -r 7_interface/requirements.txt
  python -m streamlit run 7_interface/app.py
  ```
- Send a screenshot of where it stopped. A real Codespace has not yet been started by us (we had no GitHub login that can create one). We did rehearse the same steps on 26 Oct in a clean Python 3.12 copy of the repository: the install command worked, the start command brought the app up, and the example patients gave results in a real browser.
- For running on your own computer instead, see `README.md` in this folder. Screenshots of the app are in `examples/`.
