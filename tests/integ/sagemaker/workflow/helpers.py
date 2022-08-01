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

from sagemaker.workflow.pipeline import _PipelineExecution, ImmutablePipeline


def assert_pipeline_executions(output: _PipelineExecution, expected: _PipelineExecution):
    assert output.arn == expected.arn
    assert output.pipeline == expected.pipeline


def assert_immutable_pipelines(output: ImmutablePipeline, expected: ImmutablePipeline):
    assert output.name == expected.name
    assert output.parameters == expected.parameters
    assert len(output.steps) == 0
    assert isinstance(output, ImmutablePipeline)
