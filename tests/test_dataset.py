import io
import unittest

from dataset import PolarsLogProcessor

HEADER = "PID,Timestamp,EntropyRatio,ContentOverwrite,IsDelete,DirTraversal,TypeCoverage,WriteFrequency,GroundTruth"


def processor(*rows: str, header: str = HEADER) -> PolarsLogProcessor:
    return PolarsLogProcessor(io.StringIO("\n".join([header, *rows]) + "\n"))


def row(pid: int, time: str, value: float, label: int, is_delete: str = "false") -> str:
    """Riga con tutte le feature uguali a `value`, tranne IsDelete."""
    return f"{pid},{time},{value},{value},{is_delete},{value},{value},{value},{label}"


class TestNOperations(unittest.TestCase):
    def test_splits_each_process_every_n_operations(self):
        p = processor(
            row(1, "10:00:01.000", 0.1, 0),
            row(1, "10:00:02.000", 0.2, 0),
            row(1, "10:00:03.000", 0.3, 1),
            row(2, "10:00:01.500", 0.9, 1),
        )
        windows = p.create_windows("n-operations", window_size=2)
        self.assertEqual([len(labels) for _, labels in windows], [2, 1, 1])
        features, labels = windows[0]
        self.assertEqual(list(features.shape), [2, 6])
        self.assertEqual(labels.tolist(), [0.0, 0.0])

    def test_rows_are_sorted_by_time(self):
        p = processor(row(1, "10:00:02.000", 0.2, 1), row(1, "10:00:01.000", 0.1, 0))
        features, labels = p.create_windows("n-operations", window_size=5)[0]
        self.assertAlmostEqual(features[0, 0].item(), 0.1)
        self.assertEqual(labels.tolist(), [0.0, 1.0])

    def test_is_delete_becomes_float(self):
        p = processor(row(1, "10:00:01.000", 0.0, 0, is_delete="true"))
        features, _ = p.create_windows("n-operations")[0]
        self.assertEqual(features[0, 2].item(), 1.0)


class TestTimeWindows(unittest.TestCase):
    def test_splits_by_elapsed_seconds_from_process_start(self):
        p = processor(
            row(1, "10:00:01.000", 0.1, 0),
            row(1, "10:00:03.000", 0.2, 0),
            row(1, "10:00:15.000", 0.9, 1),
        )
        windows = p.create_windows("time-windows", time_window_secs=10.0)
        self.assertEqual([labels.tolist() for _, labels in windows], [[0.0, 0.0], [1.0]])


class TestStateWindows(unittest.TestCase):
    """Finestre già calcolate con stato separato (Windowed in Progetto-Cybersecurity)."""

    HEADER = HEADER + ",WindowStart"

    def test_one_sample_per_window_even_when_overlapping(self):
        p = processor(
            row(1, "10:00:01.000", 0.1, 0) + ",0",
            row(1, "10:00:02.000", 0.2, 0) + ",0",
            row(1, "10:00:02.000", 0.5, 0) + ",1",  # stessa operazione, finestra successiva
            row(1, "10:00:03.000", 0.6, 1) + ",1",
            row(2, "10:00:01.000", 0.9, 1) + ",0",
            header=self.HEADER,
        )
        windows = p.create_windows("state-windows")
        self.assertEqual(
            [[round(v, 4) for v in features[:, 0].tolist()] for features, _ in windows],
            [[0.1, 0.2], [0.5, 0.6], [0.9]],
        )
        self.assertEqual([labels.tolist() for _, labels in windows], [[0.0, 0.0], [0.0, 1.0], [1.0]])

    def test_requires_window_start_column(self):
        p = processor(row(1, "10:00:01.000", 0.1, 0))
        with self.assertRaisesRegex(ValueError, "WindowStart"):
            p.create_windows("state-windows")

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            processor(row(1, "10:00:01.000", 0.1, 0)).create_windows("altro")


if __name__ == "__main__":
    unittest.main()
