# SPDX-FileCopyrightText: 2026 Edgar Gomes de Araujo <talktoedgar@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

import Config

# Enable the Nerves integration with Mix
Application.start(:nerves_bootstrap)

config :bb_policy_firmware, target: Mix.target()

# Use shoehorn to start the main application. See the shoehorn docs for more
# information on how to use shoehorn.
config :shoehorn, init: [:nerves_runtime, :nerves_pack]

# Set the SOURCE_DATE_EPOCH date for reproducible builds.
# See https://reproducible-builds.org/docs/source-date-epoch/ for more information

config :logger, backends: [RingLogger]

# Import target specific config. This must remain at the bottom of this file so
# it overrides the configuration defined above.
import_config "#{Mix.target()}.exs"
