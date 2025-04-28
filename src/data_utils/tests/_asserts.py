import pytest
import torch


def assert_tensors_close(a, b, rtol=1e-5, atol=1e-8):
    """Assert that tensors are close with proper handling of multi-dimensional tensors."""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        # Find values that don't match
        mask = ~torch.isclose(a, b, rtol=rtol, atol=atol)

        # Handle multi-dimensional tensors properly
        non_matching_indices = torch.nonzero(mask, as_tuple=True)

        if len(non_matching_indices[0]) > 0:
            # Get the first failing coordinate
            first_idx_tuple = tuple(idx[0].item() for idx in non_matching_indices)

            # Get values using the original coordinates (not flattened)
            expected_val = a[first_idx_tuple].item()
            actual_val = b[first_idx_tuple].item()

            # Calculate metrics
            difference = abs(expected_val - actual_val)
            allowed_tolerance = atol + rtol * abs(expected_val)

            error_msg = (
                f"Tensors not close at index {first_idx_tuple}:\n"
                f"Expected: {expected_val}\n"
                f"Actual: {actual_val}\n"
                f"Difference: {difference}\n"
                f"Allowed tolerance: {allowed_tolerance}\n"
                f"Tolerance parameters: rtol={rtol}, atol={atol}\n"
                f"Math check: {difference} {'>' if difference > allowed_tolerance else '<='} {allowed_tolerance}"
            )

            pytest.fail(error_msg)
