# SPDX-FileCopyrightText: 2026 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule Bb.PolicyTest do
  use ExUnit.Case
  doctest Bb.Policy

  test "greets the world" do
    assert Bb.Policy.hello() == :world
  end
end
