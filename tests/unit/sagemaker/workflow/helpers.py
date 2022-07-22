# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Helper methods for testing."""
from __future__ import absolute_import

from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import Step, StepTypeEnum
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.pipeline import _STEP_NAME, _OUT_BOUND_EDGES, _NEXT_STEP_NAME


def ordered(obj):
    """Helper function for dict comparison.

    Recursively orders a json-like dict or list of dicts.

    Args:
        obj: either a list or a dict

    Returns:
        either a sorted list of elements or sorted list of tuples
    """
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def equal_adjacency_list_with_edges(output, expected):
    assert len(output) == len(expected)

    output = sorted(output, key=lambda x: x["StepName"])
    expected = sorted(expected, key=lambda x: x["StepName"])

    for i in range(len(output)):
        assert output[i][_STEP_NAME] == expected[i][_STEP_NAME]
        output_outBoundEdges = sorted(output[i][_OUT_BOUND_EDGES], key=lambda x: x[_NEXT_STEP_NAME])
        expected_outBoundEdges = sorted(
            expected[i][_OUT_BOUND_EDGES], key=lambda x: x[_NEXT_STEP_NAME]
        )
        assert len(output_outBoundEdges) == len(expected_outBoundEdges)
        for j in range(len(output_outBoundEdges)):
            assert output_outBoundEdges[j] == expected_outBoundEdges[j]


class CustomStep(Step):
    def __init__(self, name, input_data=None, display_name=None, description=None, depends_on=None):
        self.input_data = input_data
        super(CustomStep, self).__init__(
            name, display_name, description, StepTypeEnum.TRAINING, depends_on
        )
        # for testing property reference, we just use DescribeTrainingJobResponse shape here.
        self._properties = Properties(name, shape_name="DescribeTrainingJobResponse")

    @property
    def arguments(self):
        if self.input_data:
            return {"input_data": self.input_data}
        return dict()

    @property
    def properties(self):
        return self._properties


class CustomStepCollection(StepCollection):
    def __init__(self, name, num_steps=2, depends_on=None):
        steps = []
        previous_step = None
        for i in range(num_steps):
            step_depends_on = depends_on if not previous_step else [previous_step]
            step = CustomStep(name=f"{name}-{i}", depends_on=step_depends_on)
            steps.append(step)
            previous_step = step
        super(CustomStepCollection, self).__init__(name, steps)
