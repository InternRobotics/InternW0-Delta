# Third-party components

`benchmark/` contains the RoboDojo evaluation runtime derived from
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo), under its
[MIT license](benchmark/LICENSE).

`benchmark/XPolicyLab/` contains the WebSocket transport, observation/action
utilities and model interface derived from
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), under its
[Apache-2.0 license](benchmark/XPolicyLab/LICENSE). The deployment adapter uses
InternW0-delta's policy implementation.

Isaac Lab (BSD-3-Clause) and cuRobo (Apache-2.0, with separately licensed assets)
are installed using `sources.json`; their licenses remain
in those checkouts. Isaac Sim, benchmark assets and model weights are obtained
separately under their respective upstream terms.
