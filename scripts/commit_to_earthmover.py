import argparse

from arraylake import Client
from fronts import utils

store_path = "/ourdisk/hpc/ai2es/tman/restructured_front_data/icechunk"
virtual_chunk_local_path = "/ourdisk/hpc/ai2es/tman/restructured_front_data/netcdf/"


def commit_to_earthmover(
    store_path: str = store_path,
    virtual_chunk_local_path: str = virtual_chunk_local_path,
    repo_name: str = "ai2es/fronts",
    branch: str = "main",
    group: str = "my-group",
    commit_message: str = "Initial Commit",
) -> None:
    """Copy the Icechunk dataset into an Arraylake repository and commit it."""
    client = Client()
    repo = client.get_repo(repo_name)
    session = repo.writable_session(branch)

    ds = utils.open_readonly_icechunk_store(
        store_path=store_path,
        branch=branch,
        virtual_chunk_local_path=virtual_chunk_local_path,
        chunks=None,
    )
    ds.to_zarr(session.store, group=group, zarr_format=3)
    session.commit(commit_message)


def main() -> None:
    """Entry point for committing the dataset to Arraylake."""
    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--store-path", default=store_path)
    parser.add_argument("--virtual-chunk-local-path", default=virtual_chunk_local_path)
    parser.add_argument("--repo-name", default="ai2es/fronts")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--group", default="my-group")
    parser.add_argument("--commit-message", default="Initial Commit")
    args = parser.parse_args()

    commit_to_earthmover(
        store_path=args.store_path,
        virtual_chunk_local_path=args.virtual_chunk_local_path,
        repo_name=args.repo_name,
        branch=args.branch,
        group=args.group,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
