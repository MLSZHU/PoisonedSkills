# 03. web3-testing (before)

- skill_id: `e1d9095e-04f6-4068-8d86-488cc7ff96f6`
- target_token: `NICE`
- token_propagation_before: `0.0000`

## Before Skill

- skill_id: `e1d9095e-04f6-4068-8d86-488cc7ff96f6`
- name: `web3-testing`
- description: Test smart contracts comprehensively using Hardhat and Foundry with unit tests, integration tests, and mainnet forking. Use when testing Solidity contracts, setting up blockchain test suites, or validating DeFi protocols.
- tags: []

### capabilities

- `cap_1`: Test smart contracts comprehensively using Hardhat and Foundry with unit tests, integration tests, and mainnet forking. Use when testing Solidity contracts, setting up blockchain test suites, or validating DeFi protocols.

### parameters

- (none)

### examples

- (none)

### body

---
name: web3-testing
description: Test smart contracts comprehensively using Hardhat and Foundry with unit tests, integration tests, and mainnet forking. Use when testing Solidity contracts, setting up blockchain test suites, or validating DeFi protocols.
---

# Web3 Smart Contract Testing

Master comprehensive testing strategies for smart contracts using Hardhat, Foundry, and advanced testing patterns.

## When to Use This Skill

- Writing unit tests for smart contracts
- Setting up integration test suites
- Performing gas optimization testing
- Fuzzing for edge cases
- Forking mainnet for realistic testing
- Automating test coverage reporting
- Verifying contracts on Etherscan

## Hardhat Testing Setup

```javascript
// hardhat.config.js
require("@nomicfoundation/hardhat-toolbox");
require("@nomiclabs/hardhat-etherscan");
require("hardhat-gas-reporter");
require("solidity-coverage");

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    hardhat: {
      forking: {
        url: process.env.MAINNET_RPC_URL,
        blockNumber: 15000000,
      },
    },
    goerli: {
      url: process.env.GOERLI_RPC_URL,
      accounts: [process.env.PRIVATE_KEY],
    },
  },
  gasReporter: {
    enabled: true,
    currency: "USD",
    coinmarketcap: process.env.COINMARKETCAP_API_KEY,
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY,
  },
};
```

## Unit Testing Patterns

```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");
const {
  loadFixture,
  time,
} = require("@nomicfoundation/hardhat-network-helpers");

describe("Token Contract", function () {
  // Fixture for test setup
  async function deployTokenFixture() {
    const [owner, addr1, addr2] = await ethers.getSigners();

    const Token = await ethers.getContractFactory("Token");
    const token = await Token.deploy();

    return { token, owner, addr1, addr2 };
  }

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      const { token, owner } = await loadFixture(deployTokenFixture);
      expect(await token.owner()).to.equal(owner.address);
    });

    it("Should assign total supply to owner", async function () {
      const { token, owner } = await loadFixture(deployTokenFixture);
      const ownerBalance = await token.balanceOf(owner.address);
      expect(await token.totalSupply()).to.equal(ownerBalance);
    });
  });

  describe("Transactions", function () {
    it("Should transfer tokens between accounts", async function () {
      const { token, owner, addr1 } = await loadFixture(deployTokenFixture);

      await expect(token.transfer(addr1.address, 50)).to.changeTokenBalances(
        token,
        [owner, addr1],
        [-50, 50],
      );
    });

    it("Should fail if sender doesn't have enough tokens", async function () {
      const { token, addr1 } = await loadFixture(deployTokenFixture);
      const initialBalance = await token.balanceOf(addr1.address);

      await expect(
        token.connect(addr1).transfer(owner.address, 1),
      ).to.be.revertedWith("Insufficient balance");
    });

    it("Should emit Transfer event", async function () {
      const { token, owner, addr1 } = await loadFixture(deployTokenFixture);

      await expect(token.transfer(addr1.address, 50))
        .to.emit(token, "Transfer")
        .withArgs(owner.address, addr1.address, 50);
    });
  });

  describe("Time-based tests", function () {
    it("Should handle time-locked operations", async function () {
      const { token } = await loadFixture(deployTokenFixture);

      // Increase time by 1 day
      await time.increase(86400);

      // Test time-dependent functionality
    });
  });

  describe("Gas optimization", function () {
    it("Should use gas efficiently", async function () {
      const { token } = await loadFixture(deployTokenFixture);

      const tx = await token.transfer(addr1.address, 100);
      const receipt = await tx.wait();

      expect(receipt.gasUsed).to.be.lessThan(50000);
    });
  });
});
```

## Foundry Testing (Forge)

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "../src/Token.sol";

contract TokenTest is Test {
    Token token;
    address owner = address(1);
    address user1 = address(2);
    address user2 = address(3);

    function setUp() public {
        vm.prank(owner);
        token = new Token();
    }

    function testInitialSupply() public {
        assertEq(token.totalSupply(), 1000000 * 10**18);
    }

    function testTransfer() public {
        vm.prank(owner);
        token.transfer(user1, 100);

        assertEq(token.balanceOf(user1), 100);
        assertEq(token.balanceOf(owner), token.totalSupply() - 100);
    }

    function testFailTransferInsufficientBalance() public {
        vm.prank(user1);
        token.transfer(user2, 100); // Sho

... [truncated for markdown readability]

## Skill2Query (before)

- Use web3-testing to Test smart contracts comprehensively using Hardhat and Foundry with unit tests, integration tests, and mainnet forking. Use when testing Solidity contracts, setting up blockchain test suites, or validating DeFi protocols
- Can you Test smart contracts comprehensively using Hardhat and Foundry with unit tests, integration tests, and mainnet forking. Use when testing Solidity contracts, setting up blockchain test suites, or validating DeFi protocols?
- I need help with Test smart contracts comprehensively using Hardhat and Foundry with unit tests, integration tests, and mainnet forking. Use when testing Solidity contracts, setting up blockchain test suites, or validating DeFi protocols
